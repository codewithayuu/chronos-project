"""
Voice Alert Formatter — Formats text for TTS (Sarvam AI / browser).

Produces clean, naturally-paced clinical text optimized for
text-to-speech systems.

Returns structured output:
  - voice_text: clean text for TTS
  - ssml_hints: pacing/emphasis markers
  - priority: urgency level for voice queue
  - language: target language code
"""

from typing import Dict, Any, Optional
from ..models import PatientState, AlertSeverity


class VoiceFormatter:
    """Formats patient alerts and narratives for voice synthesis."""

    def format_alert(
        self, state: PatientState, narrative_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format a voice alert for a patient.

        Parameters
        ----------
        state : PatientState
        narrative_text : optional full narrative from NarrativeGenerator

        Returns
        -------
        Dict with voice_text, priority, sections for TTS consumption.
        """
        severity = state.alert.severity
        pid_spoken = self._speak_patient_id(state.patient_id)

        if severity == AlertSeverity.CRITICAL:
            return self._critical_alert(state, pid_spoken)
        elif severity == AlertSeverity.WARNING:
            return self._warning_alert(state, pid_spoken)
        elif severity == AlertSeverity.WATCH:
            return self._watch_alert(state, pid_spoken)
        else:
            return self._status_update(
                state, pid_spoken, narrative_text
            )

    def _critical_alert(
        self, state: PatientState, pid: str
    ) -> Dict:
        """Format critical voice alert."""
        ces = state.composite_entropy
        msg = state.alert.message or "Critical deterioration"

        # Find worst vital
        worst = self._find_worst_vital(state)
        worst_text = (
            f"Primary concern: {worst}. "
            if worst
            else ""
        )

        # Drug masking warning
        drug_text = ""
        if state.alert.drug_masked:
            drug_text = (
                "Warning: active medications may be masking "
                "the true severity. "
            )

        # Time estimate
        time_text = ""
        if state.alert.hours_to_predicted_event is not None:
            hrs = state.alert.hours_to_predicted_event
            if hrs < 1:
                time_text = (
                    f"Estimated time to critical event: "
                    f"{int(hrs * 60)} minutes. "
                )
            else:
                time_text = (
                    f"Estimated time to critical event: "
                    f"{hrs:.1f} hours. "
                )

        voice = (
            f"Critical alert. {pid}. "
            f"Entropy score {ces:.2f}. "
            f"{worst_text}"
            f"{drug_text}"
            f"{time_text}"
            f"Immediate bedside assessment required."
        )

        return {
            "voice_text": voice,
            "priority": "critical",
            "severity": "CRITICAL",
            "patient_id": state.patient_id,
            "language": "en-IN",
            "rate": 0.85,
            "pitch": 0.7,
        }

    def _warning_alert(
        self, state: PatientState, pid: str
    ) -> Dict:
        """Format warning voice alert."""
        ces = state.composite_entropy
        worst = self._find_worst_vital(state)
        worst_text = (
            f"Declining entropy in {worst}. "
            if worst
            else ""
        )

        voice = (
            f"Warning. {pid}. "
            f"Entropy score declining to {ces:.2f}. "
            f"{worst_text}"
            f"Clinical evaluation recommended."
        )

        return {
            "voice_text": voice,
            "priority": "warning",
            "severity": "WARNING",
            "patient_id": state.patient_id,
            "language": "en-IN",
            "rate": 0.9,
            "pitch": 0.8,
        }

    def _watch_alert(
        self, state: PatientState, pid: str
    ) -> Dict:
        """Format watch-level voice alert."""
        ces = state.composite_entropy

        voice = (
            f"Attention. {pid}. "
            f"Entropy score is {ces:.2f}. "
            f"Increased monitoring advised."
        )

        return {
            "voice_text": voice,
            "priority": "watch",
            "severity": "WATCH",
            "patient_id": state.patient_id,
            "language": "en-IN",
            "rate": 0.95,
            "pitch": 0.9,
        }

    def _status_update(
        self, state: PatientState, pid: str, narrative: str = None
    ) -> Dict:
        """Format a routine status update."""
        ces = state.composite_entropy

        if narrative:
            # Use first 200 chars of narrative
            short = narrative[:200]
            if len(narrative) > 200:
                short += "..."
            voice = f"{pid} status update. {short}"
        else:
            voice = (
                f"{pid}. Status stable. "
                f"Entropy score {ces:.2f}. "
                f"No concerns at this time."
            )

        return {
            "voice_text": voice,
            "priority": "info",
            "severity": "NONE",
            "patient_id": state.patient_id,
            "language": "en-IN",
            "rate": 1.0,
            "pitch": 1.0,
        }

    def _speak_patient_id(self, patient_id: str) -> str:
        """Convert patient ID to speakable form."""
        # "HERO-SEPSIS-001" -> "Patient Hero Sepsis 001"
        clean = patient_id.replace("-", " ").replace("_", " ")
        return f"Patient {clean}"

    def _find_worst_vital(
        self, state: PatientState
    ) -> Optional[str]:
        """Find the vital sign with lowest entropy."""
        worst_name = None
        worst_entropy = 1.0

        for name in [
            "heart_rate", "spo2", "bp_systolic",
            "resp_rate", "temperature",
        ]:
            detail = getattr(state.vitals, name, None)
            if (
                detail
                and detail.sampen_normalized is not None
                and detail.sampen_normalized < worst_entropy
            ):
                worst_entropy = detail.sampen_normalized
                worst_name = name.replace("_", " ")

        return worst_name
