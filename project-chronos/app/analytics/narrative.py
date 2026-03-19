"""
Clinical Narrative Generator — template-based, deterministic.

Generates a plain-English clinical summary from PatientState.
No LLM needed. No API keys. Works offline in Docker.

This makes the system self-explanatory: when a judge asks
'Why is this patient red?', the system answers in clinical English.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from ..models import PatientState, AlertSeverity, TrendDirection


class NarrativeGenerator:
    """Generates human-readable clinical summaries from patient state."""

    def generate(
        self,
        state: PatientState,
        history_length_minutes: int = 0,
        decoupling_summary: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured clinical narrative.

        Parameters
        ----------
        state : PatientState
            Current patient state with all analytics
        history_length_minutes : int
            How long this patient has been monitored
        decoupling_summary : dict or None
            Output from CrossVitalAnalyzer.get_decoupling_summary()

        Returns
        -------
        Dict with sections, full_text, severity, and generated_at.
        """
        sections = {}

        sections["overview"] = self._overview(state, history_length_minutes)
        sections["entropy_analysis"] = self._entropy_analysis(state)
        sections["vital_assessment"] = self._vital_assessment(state)
        sections["drug_context"] = self._drug_context(state)
        sections["organ_coupling"] = self._organ_coupling(decoupling_summary)
        sections["clinical_scores"] = self._clinical_scores(state)
        sections["recommendations"] = self._recommendations(state)
        sections["risk_assessment"] = self._risk_assessment(state)

        # Filter out empty sections
        sections = {k: v for k, v in sections.items() if v}

        full_text = "\n\n".join(sections.values())

        return {
            "patient_id": state.patient_id,
            "generated_at": datetime.utcnow().isoformat(),
            "severity": state.alert.severity.value,
            "sections": sections,
            "full_text": full_text,
        }

    def _overview(self, state: PatientState, history_minutes: int) -> str:
        hours = history_minutes / 60 if history_minutes > 0 else 0
        time_str = f" over the past {hours:.1f} hours" if hours > 0 else ""

        if state.calibrating:
            return (
                f"Patient {state.patient_id} -- System is collecting "
                f"baseline data. Entropy analysis will begin after "
                f"{state.window_fill:.0%} of the calibration window "
                f"is filled."
            )

        severity_text = {
            AlertSeverity.NONE: (
                "currently stable with healthy physiological complexity"
            ),
            AlertSeverity.WATCH: (
                "showing early signs of reduced physiological complexity"
            ),
            AlertSeverity.WARNING: (
                "exhibiting significant loss of physiological complexity"
            ),
            AlertSeverity.CRITICAL: (
                "in critical condition with severe loss of "
                "physiological variability"
            ),
        }

        desc = severity_text.get(
            state.alert.severity, "being monitored"
        )

        return (
            f"Patient {state.patient_id}{time_str} is {desc}. "
            f"Composite Entropy Score: "
            f"{state.composite_entropy:.2f}/1.00."
        )

    def _entropy_analysis(self, state: PatientState) -> str:
        if state.calibrating:
            return ""

        # Find vitals with declining entropy
        declining = []
        vital_names = [
            "heart_rate", "spo2", "bp_systolic",
            "bp_diastolic", "resp_rate", "temperature",
        ]
        for name in vital_names:
            detail = getattr(state.vitals, name, None)
            if detail is None:
                continue
            if (
                detail.trend == TrendDirection.FALLING
                and detail.sampen_normalized is not None
            ):
                display_name = name.replace("_", " ").title()
                declining.append(
                    (display_name, detail.sampen_normalized)
                )

        if not declining:
            return (
                "All vital sign entropy levels are within normal "
                "ranges. Signal complexity is maintained across "
                "all monitored parameters, indicating healthy "
                "autonomic regulation."
            )

        vital_list = ", ".join(
            f"{name} (entropy: {val:.2f})"
            for name, val in declining
        )

        text = (
            f"Entropy decline detected in: {vital_list}. "
            f"This indicates increasing signal regularity — "
            f"the body's normal beat-to-beat and breath-to-breath "
            f"variability is diminishing. This loss of complexity "
            f"suggests reduced autonomic regulatory capacity."
        )

        # Note if raw vs adjusted CES differ
        raw = state.composite_entropy_raw
        adj = state.composite_entropy
        if abs(raw - adj) > 0.01:
            text += (
                f" Note: Raw CES ({raw:.2f}) has been adjusted "
                f"to {adj:.2f} after accounting for active "
                f"medication effects."
            )

        return text

    def _vital_assessment(self, state: PatientState) -> str:
        if state.calibrating:
            return ""

        vitals_info = [
            ("heart_rate", "Heart Rate", "bpm"),
            ("spo2", "SpO2", "%"),
            ("bp_systolic", "Systolic BP", "mmHg"),
            ("bp_diastolic", "Diastolic BP", "mmHg"),
            ("resp_rate", "Respiratory Rate", "/min"),
            ("temperature", "Temperature", "C"),
        ]

        vitals_text = []
        for attr, label, unit in vitals_info:
            detail = getattr(state.vitals, attr, None)
            if detail and detail.value is not None:
                vitals_text.append(
                    f"{label}: {detail.value:.0f} {unit}"
                )

        if not vitals_text:
            return ""

        intro = "Current vital signs: " + ", ".join(vitals_text) + "."

        # The key insight: normal values but low entropy
        normal_ranges = {
            "heart_rate": (50, 120),
            "spo2": (90, 100),
            "bp_systolic": (90, 180),
            "resp_rate": (8, 30),
        }
        all_normal = True
        for attr, (low, high) in normal_ranges.items():
            detail = getattr(state.vitals, attr, None)
            if detail and detail.value is not None:
                if detail.value < low or detail.value > high:
                    all_normal = False

        if (
            all_normal
            and state.alert.severity
            in (AlertSeverity.WARNING, AlertSeverity.CRITICAL)
        ):
            intro += (
                " IMPORTANT: All vital sign VALUES are within "
                "normal ranges. However, PATTERNS of these "
                "signals have become abnormally regular. "
                "Traditional threshold-based monitors would NOT "
                "alarm at this time. Entropy analysis reveals "
                "underlying physiological deterioration that "
                "precedes visible vital sign changes by hours."
            )

        return intro

    def _drug_context(self, state: PatientState) -> str:
        if not state.active_drugs:
            return (
                "No active medications influencing vital sign "
                "interpretation."
            )

        drug_lines = []
        for drug in state.active_drugs:
            line = f"{drug.drug_name}"
            if drug.dose and drug.unit:
                line += f" ({drug.dose} {drug.unit})"
            if drug.drug_class:
                line += f" [{drug.drug_class}]"
            drug_lines.append(line)

        text = (
            "Active medications: " + ", ".join(drug_lines) + "."
        )

        if state.alert.drug_masked:
            text += (
                " DRUG MASKING DETECTED: One or more medications "
                "may be artificially stabilizing vital sign values "
                "while underlying physiological complexity "
                "continues to decline. The true patient condition "
                "may be worse than vital signs suggest. "
                "Clinical correlation is strongly recommended."
            )

        return text

    def _organ_coupling(
        self, decoupling_summary: Optional[Dict]
    ) -> str:
        if decoupling_summary is None:
            return ""

        count = decoupling_summary.get("decoupled_count", 0)
        total = decoupling_summary.get("total_pairs", 0)

        if count == 0:
            return (
                "Inter-vital correlations are maintained. "
                "Organ systems remain coupled, indicating "
                "coordinated physiological regulation."
            )

        alert = decoupling_summary.get("clinical_alert", "")
        pairs = decoupling_summary.get("decoupled_pairs", [])
        systems = decoupling_summary.get("affected_systems", [])

        text = (
            f"ORGAN SYSTEM DECOUPLING: {count}/{total} vital sign "
            f"correlations have broken down. Decoupled pairs: "
            f"{', '.join(pairs)}. Affected organ systems: "
            f"{', '.join(systems)}. "
        )

        if count >= 3:
            text += (
                "This pattern of multi-system decoupling is "
                "associated with impending multi-organ failure "
                "and should trigger immediate clinical review."
            )
        elif count >= 2:
            text += (
                "Multi-system involvement detected. Recommend "
                "close monitoring for progression to "
                "multi-organ dysfunction."
            )
        else:
            text += (
                "Single-system involvement. Monitor for "
                "progression."
            )

        return text

    def _clinical_scores(self, state: PatientState) -> str:
        if state.clinical_scores is None:
            return ""

        news2 = state.clinical_scores.get("news2", {})
        qsofa = state.clinical_scores.get("qsofa", {})
        n_score = news2.get("score", 0)
        n_risk = news2.get("risk_level", "Unknown")
        q_score = qsofa.get("score", 0)

        text = (
            f"Standard clinical scores: NEWS2 = {n_score}/20 "
            f"({n_risk} risk), qSOFA = {q_score}/3."
        )

        severity = state.alert.severity
        if (
            severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL)
            and n_risk in ("None", "Low")
        ):
            text += (
                f" NOTE: NEWS2 classifies this patient as "
                f"{n_risk} risk, yet Chronos entropy analysis "
                f"indicates {severity.value} status. This "
                f"demonstrates the gap between threshold-based "
                f"scoring and pattern-based analysis. Chronos "
                f"detects deterioration that standard scores miss."
            )

        return text

    def _recommendations(self, state: PatientState) -> str:
        if not state.interventions:
            if state.alert.severity == AlertSeverity.NONE:
                return (
                    "No interventions recommended at this time. "
                    "Continue routine monitoring."
                )
            return (
                "Clinical judgment advised. Insufficient similar "
                "historical cases for specific recommendations."
            )

        lines = [
            "Evidence-based interventions "
            "(from historical case matching):"
        ]
        for intv in state.interventions[:3]:
            lines.append(
                f"  {intv.rank}. {intv.action} -- "
                f"{intv.historical_success_rate:.0%} success rate "
                f"(n={intv.similar_cases_count} similar cases, "
                f"median response: "
                f"{intv.median_response_time_hours}h)"
            )

        return "\n".join(lines)

    def _risk_assessment(self, state: PatientState) -> str:
        if state.calibrating:
            return ""

        severity = state.alert.severity
        time_str = ""
        if state.alert.hours_to_predicted_event is not None:
            hrs = state.alert.hours_to_predicted_event
            time_str = (
                f" Estimated time to critical event: "
                f"{hrs:.1f} hours."
            )

        assessments = {
            AlertSeverity.NONE: (
                "RISK LEVEL: LOW. Continue routine monitoring."
            ),
            AlertSeverity.WATCH: (
                f"RISK LEVEL: MODERATE. Increased surveillance "
                f"recommended. Consider bedside assessment."
                f"{time_str}"
            ),
            AlertSeverity.WARNING: (
                f"RISK LEVEL: HIGH. Active clinical evaluation "
                f"recommended. Review intervention options."
                f"{time_str}"
            ),
            AlertSeverity.CRITICAL: (
                f"RISK LEVEL: CRITICAL. Immediate bedside "
                f"assessment required. Prepare for emergency "
                f"intervention.{time_str}"
            ),
        }

        return assessments.get(severity, "Risk assessment unavailable.")
