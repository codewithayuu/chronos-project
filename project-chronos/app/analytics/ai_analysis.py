"""
AI-Enhanced Analysis — Grok/Gemini integration with fallback.

Sends patient state to an LLM for enhanced clinical analysis.
Falls back to template-based narrative if no API key or on error.

Supports:
  - Google Gemini (GEMINI_API_KEY env var)
  - xAI Grok (GROK_API_KEY env var)
  - Fallback to NarrativeGenerator (always works)
"""

import os
import json
import logging
from typing import Dict, Any, Optional

from ..models import PatientState

logger = logging.getLogger(__name__)


def _build_prompt(state: PatientState, extra_context: Dict = None) -> str:
    """Build a clinical analysis prompt from patient state."""
    vitals_text = []
    for name in [
        "heart_rate", "spo2", "bp_systolic",
        "bp_diastolic", "resp_rate", "temperature",
    ]:
        detail = getattr(state.vitals, name, None)
        if detail and detail.value is not None:
            entropy_str = ""
            if detail.sampen_normalized is not None:
                entropy_str = (
                    f", entropy: {detail.sampen_normalized:.2f}/1.0"
                )
            trend_str = (
                f", trend: {detail.trend.value}"
                if detail.trend
                else ""
            )
            display = name.replace("_", " ").title()
            vitals_text.append(
                f"  {display}: {detail.value:.1f}"
                f"{entropy_str}{trend_str}"
            )

    vitals_block = "\n".join(vitals_text) if vitals_text else "  No data"

    drugs_text = "None"
    if state.active_drugs:
        drug_list = []
        for d in state.active_drugs:
            s = d.drug_name
            if d.dose and d.unit:
                s += f" ({d.dose} {d.unit})"
            drug_list.append(s)
        drugs_text = ", ".join(drug_list)

    scores_text = "Not computed"
    if state.clinical_scores:
        news2 = state.clinical_scores.get("news2", {})
        qsofa = state.clinical_scores.get("qsofa", {})
        scores_text = (
            f"NEWS2={news2.get('score', '?')} "
            f"({news2.get('risk_level', '?')}), "
            f"qSOFA={qsofa.get('score', '?')} "
            f"({qsofa.get('risk_level', '?')})"
        )

    decoupling_text = "Not analyzed"
    if extra_context and "decoupling" in extra_context:
        dec = extra_context["decoupling"]
        decoupling_text = (
            f"{dec.get('decoupled_count', 0)}/"
            f"{dec.get('total_pairs', 5)} organ couplings "
            f"disrupted"
        )
        if dec.get("clinical_alert"):
            decoupling_text += f"\n  {dec['clinical_alert']}"

    prompt = f"""You are a senior ICU clinical decision support system. Analyze this patient's data and provide a concise clinical assessment.

PATIENT: {state.patient_id}
COMPOSITE ENTROPY SCORE: {state.composite_entropy:.3f}/1.000 (lower = more dangerous)
ALERT SEVERITY: {state.alert.severity.value}
DRUG MASKING: {"YES - medications may be hiding deterioration" if state.alert.drug_masked else "No"}

VITAL SIGNS (with entropy analysis):
{vitals_block}

ACTIVE MEDICATIONS: {drugs_text}
CLINICAL SCORES: {scores_text}
ORGAN COUPLING: {decoupling_text}

KEY CONTEXT:
- Entropy measures physiological complexity. Healthy patients have HIGH entropy (complex, adaptive signals). Deteriorating patients have LOW entropy (rigid, predictable patterns).
- Entropy decline often precedes vital sign changes by 2-6 hours.
- Drug masking means medications may be stabilizing VALUES while underlying PATTERNS deteriorate.

Provide:
1. CLINICAL ASSESSMENT (2-3 sentences): What is happening to this patient?
2. KEY CONCERNS (bullet points): What should the clinician worry about?
3. RECOMMENDED ACTIONS (numbered list): What should be done now?
4. RISK TRAJECTORY: Is this patient likely to improve, stabilize, or deteriorate?

Be specific, actionable, and concise. Use clinical terminology appropriate for an ICU physician."""

    return prompt


def analyze_with_gemini(
    state: PatientState,
    extra_context: Dict = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Call Google Gemini API for enhanced analysis."""
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return {"error": "No GEMINI_API_KEY configured"}

    prompt = _build_prompt(state, extra_context)

    try:
        import urllib.request

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/gemini-2.0-flash:generateContent?key={key}"
        )
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 800,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())

        text = (
            result.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )

        return {
            "source": "gemini",
            "model": "gemini-2.0-flash",
            "analysis": text,
            "patient_id": state.patient_id,
            "severity": state.alert.severity.value,
        }
    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}")
        return {"error": f"Gemini API error: {str(e)}"}


def analyze_with_grok(
    state: PatientState,
    extra_context: Dict = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Call xAI Grok API for enhanced analysis."""
    key = api_key or os.environ.get("GROK_API_KEY", "")
    if not key:
        return {"error": "No GROK_API_KEY configured"}

    prompt = _build_prompt(state, extra_context)

    try:
        import urllib.request

        url = "https://api.x.ai/v1/chat/completions"
        payload = {
            "model": "grok-3-mini",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a senior ICU clinical decision "
                        "support AI. Be concise, specific, and "
                        "actionable."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 800,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())

        text = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        return {
            "source": "grok",
            "model": "grok-3-mini",
            "analysis": text,
            "patient_id": state.patient_id,
            "severity": state.alert.severity.value,
        }
    except Exception as e:
        logger.warning(f"Grok API call failed: {e}")
        return {"error": f"Grok API error: {str(e)}"}


def analyze_patient(
    state: PatientState,
    extra_context: Dict = None,
    preferred_provider: str = "auto",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze patient with best available AI provider.

    Parameters
    ----------
    state : PatientState
    extra_context : dict with decoupling info etc.
    preferred_provider : 'gemini', 'grok', or 'auto'
    api_key : explicit API key (overrides env var)

    Returns
    -------
    Dict with source, analysis text, and metadata.
    Falls back to template narrative on any failure.
    """
    result = None

    if preferred_provider in ("gemini", "auto"):
        result = analyze_with_gemini(
            state, extra_context, api_key
        )
        if "error" not in result:
            return result

    if preferred_provider in ("grok", "auto"):
        result = analyze_with_grok(
            state, extra_context, api_key
        )
        if "error" not in result:
            return result

    # Fallback to template narrative
    from .narrative import NarrativeGenerator

    gen = NarrativeGenerator()
    decoupling = (
        extra_context.get("decoupling")
        if extra_context
        else None
    )
    narrative = gen.generate(
        state, decoupling_summary=decoupling
    )

    return {
        "source": "template",
        "model": "chronos-narrative-v1",
        "analysis": narrative.get("full_text", ""),
        "patient_id": state.patient_id,
        "severity": state.alert.severity.value,
        "note": (
            "AI analysis unavailable. Using built-in "
            "clinical narrative engine."
        ),
        "ai_error": (
            result.get("error") if result else "No API key"
        ),
    }
