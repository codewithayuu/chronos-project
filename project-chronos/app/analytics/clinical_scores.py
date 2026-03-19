"""
Clinical Scoring Systems — NEWS2 and qSOFA.

Standard clinical tools computed from vital signs. Used by every
hospital worldwide. Computing these alongside entropy allows
direct comparison: Chronos detects deterioration BEFORE these
validated scores escalate.

NEWS2 (National Early Warning Score 2):
  - Score 0-20 from 7 physiological parameters
  - Used in UK NHS and internationally
  - Triggers clinical response at score >= 5

qSOFA (Quick Sequential Organ Failure Assessment):
  - Score 0-3 from RR, BP, mental status
  - Screening tool for sepsis
  - Score >= 2 suggests organ dysfunction
"""

from typing import Optional, Dict, Any


class ClinicalScores:
    """Compute standard clinical scoring systems from vital signs."""

    @staticmethod
    def compute_news2(
        hr: Optional[float] = None,
        rr: Optional[float] = None,
        spo2: Optional[float] = None,
        bp_sys: Optional[float] = None,
        temp: Optional[float] = None,
        on_supplemental_o2: bool = False,
        consciousness: str = "alert",
    ) -> Dict[str, Any]:
        """
        Compute NEWS2 score from vital signs.

        Parameters
        ----------
        hr : Heart rate (bpm)
        rr : Respiratory rate (breaths/min)
        spo2 : Oxygen saturation (%)
        bp_sys : Systolic blood pressure (mmHg)
        temp : Temperature (Celsius)
        on_supplemental_o2 : Whether patient is on supplemental O2
        consciousness : "alert", "voice", "pain", or "unresponsive" (AVPU)

        Returns
        -------
        Dict with score, risk_level, subscores, and max_possible.
        """
        score = 0
        subscores = {}

        # Respiratory Rate (0-3)
        if rr is not None:
            if rr <= 8:
                s = 3
            elif rr <= 11:
                s = 1
            elif rr <= 20:
                s = 0
            elif rr <= 24:
                s = 2
            else:
                s = 3
            score += s
            subscores["resp_rate"] = s

        # SpO2 Scale 1 (not on supplemental O2)
        if spo2 is not None:
            if not on_supplemental_o2:
                if spo2 <= 91:
                    s = 3
                elif spo2 <= 93:
                    s = 2
                elif spo2 <= 95:
                    s = 1
                else:
                    s = 0
            else:
                # Scale 2 (on supplemental O2)
                if spo2 <= 83:
                    s = 3
                elif spo2 <= 85:
                    s = 2
                elif spo2 <= 87:
                    s = 1
                elif spo2 <= 92:
                    s = 0
                elif spo2 <= 94:
                    s = 1
                elif spo2 <= 96:
                    s = 2
                else:
                    s = 3
            score += s
            subscores["spo2"] = s

        # Supplemental O2 (0-2)
        if on_supplemental_o2:
            s = 2
            score += s
            subscores["supplemental_o2"] = s

        # Systolic Blood Pressure (0-3)
        if bp_sys is not None:
            if bp_sys <= 90:
                s = 3
            elif bp_sys <= 100:
                s = 2
            elif bp_sys <= 110:
                s = 1
            elif bp_sys <= 219:
                s = 0
            else:
                s = 3
            score += s
            subscores["bp_systolic"] = s

        # Heart Rate (0-3)
        if hr is not None:
            if hr <= 40:
                s = 3
            elif hr <= 50:
                s = 1
            elif hr <= 90:
                s = 0
            elif hr <= 110:
                s = 1
            elif hr <= 130:
                s = 2
            else:
                s = 3
            score += s
            subscores["heart_rate"] = s

        # Consciousness (0-3)
        consciousness_lower = consciousness.lower() if consciousness else "alert"
        if consciousness_lower == "alert":
            s = 0
        elif consciousness_lower == "voice":
            s = 3
        elif consciousness_lower == "pain":
            s = 3
        elif consciousness_lower == "unresponsive":
            s = 3
        else:
            s = 0
        score += s
        subscores["consciousness"] = s

        # Temperature (0-3)
        if temp is not None:
            if temp <= 35.0:
                s = 3
            elif temp <= 36.0:
                s = 1
            elif temp <= 38.0:
                s = 0
            elif temp <= 39.0:
                s = 1
            else:
                s = 2
            score += s
            subscores["temperature"] = s

        # Risk classification
        # Check for any single parameter scoring 3
        has_extreme = any(v == 3 for v in subscores.values())

        if score >= 7:
            risk = "High"
            clinical_response = "Emergency response - immediate senior clinician review"
        elif score >= 5 or has_extreme:
            risk = "Medium"
            clinical_response = "Urgent response - clinician review within 1 hour"
        elif score >= 1:
            risk = "Low"
            clinical_response = "Assessment by competent ward nurse"
        else:
            risk = "None"
            clinical_response = "Routine monitoring"

        return {
            "score": score,
            "max_possible": 20,
            "risk_level": risk,
            "clinical_response": clinical_response,
            "subscores": subscores,
            "has_extreme_parameter": has_extreme,
        }

    @staticmethod
    def compute_qsofa(
        rr: Optional[float] = None,
        bp_sys: Optional[float] = None,
        altered_mental: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute qSOFA score for sepsis screening.

        Parameters
        ----------
        rr : Respiratory rate (breaths/min)
        bp_sys : Systolic blood pressure (mmHg)
        altered_mental : Whether patient has altered mental status

        Returns
        -------
        Dict with score, risk_level, criteria_met, and interpretation.
        """
        score = 0
        criteria = {}

        if rr is not None and rr >= 22:
            score += 1
            criteria["respiratory_rate_>=22"] = True
        else:
            criteria["respiratory_rate_>=22"] = False

        if bp_sys is not None and bp_sys <= 100:
            score += 1
            criteria["systolic_bp_<=100"] = True
        else:
            criteria["systolic_bp_<=100"] = False

        if altered_mental:
            score += 1
            criteria["altered_mental_status"] = True
        else:
            criteria["altered_mental_status"] = False

        if score >= 2:
            risk = "High"
            interpretation = "qSOFA >= 2: Associated with poor outcomes. Consider sepsis workup."
        elif score == 1:
            risk = "Moderate"
            interpretation = "qSOFA = 1: One criterion met. Monitor closely."
        else:
            risk = "Low"
            interpretation = "qSOFA = 0: No sepsis criteria met on quick screen."

        return {
            "score": score,
            "max_possible": 3,
            "risk_level": risk,
            "interpretation": interpretation,
            "criteria_met": criteria,
        }

    @staticmethod
    def compute_all(
        hr: Optional[float] = None,
        rr: Optional[float] = None,
        spo2: Optional[float] = None,
        bp_sys: Optional[float] = None,
        temp: Optional[float] = None,
        on_supplemental_o2: bool = False,
        consciousness: str = "alert",
        altered_mental: bool = False,
    ) -> Dict[str, Any]:
        """Compute all clinical scores from vital signs."""
        return {
            "news2": ClinicalScores.compute_news2(
                hr=hr, rr=rr, spo2=spo2, bp_sys=bp_sys,
                temp=temp, on_supplemental_o2=on_supplemental_o2,
                consciousness=consciousness,
            ),
            "qsofa": ClinicalScores.compute_qsofa(
                rr=rr, bp_sys=bp_sys, altered_mental=altered_mental,
            ),
        }
