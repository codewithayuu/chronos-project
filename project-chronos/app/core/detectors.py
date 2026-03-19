"""
Detector Bank — Core Runtime for Project Chronos.

Implements 8 consolidated clinical detectors that run on every pipeline
cycle and flag specific clinical patterns for the frontend.

Detectors:
  1. entropy_threshold  — CES below severity thresholds
  2. silent_decline     — Entropy falling while vitals appear normal
  3. drug_masking       — Active drugs may mask deterioration
  4. respiratory_risk   — Respiratory-specific deterioration pattern
  5. hemodynamic        — Hemodynamic instability pattern
  6. alarm_suppression  — Traditional alarm suppressed by drug context
  7. recovery           — Patient recovery trend detected
  8. data_quality       — Missing or unreliable data flagged
"""

from typing import Dict, List, Optional, Any


# Traditional ICU alarm thresholds (for Detector 6)
TRADITIONAL_THRESHOLDS = {
    "hr": {"low": 50, "high": 120},
    "bp_sys": {"low": 90, "high": 180},
    "bp_dia": {"low": 50, "high": 110},
    "rr": {"low": 8, "high": 30},
    "spo2": {"low": 90, "high": None},
    "temp": {"low": 35.0, "high": 39.5},
}

# Display name mapping
DETECTOR_DISPLAY_NAMES = {
    "entropy_threshold": "Entropy Threshold",
    "silent_decline": "Silent Decline",
    "drug_masking": "Drug Masking",
    "respiratory_risk": "Respiratory Risk",
    "hemodynamic": "Hemodynamic",
    "alarm_suppression": "Alarm Suppression",
    "recovery": "Recovery",
    "data_quality": "Data Quality",
}


def _make_result(
    detector_name: str,
    active: bool = False,
    severity: str = "NONE",
    message: str = "",
    contributing_factors: Optional[List[str]] = None,
    recommended_action: str = "",
) -> dict:
    """Helper to create a consistently shaped detector result dict."""
    return {
        "detector_name": detector_name,
        "active": active,
        "severity": severity,
        "message": message,
        "contributing_factors": contributing_factors or [],
        "recommended_action": recommended_action,
    }


class DetectorBank:
    """
    Bank of 8 clinical detectors.

    Each detector inspects the full patient state (entropy, drugs, vitals,
    ML predictions, fusion output) and returns a structured result dict.
    
    ``run_all()`` invokes all 8 and returns the full list.
    """

    def __init__(self):
        self.suppressed_alarm_count: Dict[str, int] = {}  # per-patient

    def run_all(
        self,
        entropy_state: dict,
        drug_state: dict,
        vitals: dict,
        ml_predictions: dict,
        fusion: dict,
    ) -> List[dict]:
        """
        Run all 8 detectors and return their results.

        Parameters
        ----------
        entropy_state : dict
            Keys: ces_adjusted, ces_slope_6h, warmup_complete,
                  sampen_hr, sampen_bp_sys, sampen_rr, sampen_spo2
        drug_state : dict
            Keys: drug_masking, active_drugs (list of drug dicts)
        vitals : dict
            Keys: hr, bp_sys, bp_dia, rr, spo2, temp
        ml_predictions : dict
            Keys: deterioration (dict or None), syndrome (dict or None)
        fusion : dict
            Keys: final_severity, final_risk_score, component_risks, etc.

        Returns
        -------
        list of dict
            8 detector results (one per detector), including inactive ones.
        """
        return [
            self._detect_entropy_threshold(entropy_state),
            self._detect_silent_decline(entropy_state, vitals),
            self._detect_drug_masking(entropy_state, drug_state, vitals),
            self._detect_respiratory_risk(entropy_state, vitals, ml_predictions),
            self._detect_hemodynamic(entropy_state, vitals, ml_predictions),
            self._detect_alarm_suppression(vitals, drug_state),
            self._detect_recovery(entropy_state, ml_predictions, fusion),
            self._detect_data_quality(entropy_state, vitals),
        ]

    # ──────────────────────────────────────────────
    # Detector 1: Entropy Threshold
    # ──────────────────────────────────────────────

    def _detect_entropy_threshold(self, entropy_state: dict) -> dict:
        """
        Fires when CES drops below severity thresholds.
        """
        ces = entropy_state.get("ces_adjusted", 1.0)
        warmup = entropy_state.get("warmup_complete", False)

        if not warmup:
            return _make_result("entropy_threshold")

        if ces < 0.20:
            return _make_result(
                "entropy_threshold",
                active=True,
                severity="CRITICAL",
                message=f"CES at CRITICAL level ({ces:.2f})",
                contributing_factors=[f"ces_adjusted: {ces:.2f}"],
                recommended_action="Immediate clinical review required",
            )
        elif ces < 0.40:
            return _make_result(
                "entropy_threshold",
                active=True,
                severity="WARNING",
                message=f"CES below WARNING threshold ({ces:.2f})",
                contributing_factors=[f"ces_adjusted: {ces:.2f}"],
                recommended_action="Monitor closely",
            )
        elif ces < 0.60:
            return _make_result(
                "entropy_threshold",
                active=True,
                severity="WATCH",
                message=f"CES below WATCH threshold ({ces:.2f})",
                contributing_factors=[f"ces_adjusted: {ces:.2f}"],
                recommended_action="Continue monitoring",
            )

        return _make_result("entropy_threshold")

    # ──────────────────────────────────────────────
    # Detector 2: Silent Decline
    # ──────────────────────────────────────────────

    def _detect_silent_decline(self, entropy_state: dict, vitals: dict) -> dict:
        """
        Fires when entropy is declining but vital VALUES remain in normal range.
        This is the key 'early warning' detector — the entire premise of Chronos.
        """
        ces_slope = entropy_state.get("ces_slope_6h", 0.0)
        warmup = entropy_state.get("warmup_complete", False)

        if not warmup or ces_slope >= -0.001:
            return _make_result("silent_decline")

        # Check if all vitals are in normal range
        all_normal = self._are_vitals_normal(vitals)

        if all_normal and ces_slope < -0.001:
            severity = "CRITICAL" if ces_slope < -0.003 else "WARNING"
            return _make_result(
                "silent_decline",
                active=True,
                severity=severity,
                message="Silent decline — entropy falling while vitals appear normal",
                contributing_factors=[
                    f"ces_slope_6h: {ces_slope:.4f}",
                    "all_vitals_in_range: true",
                ],
                recommended_action="Review patient at bedside",
            )

        return _make_result("silent_decline")

    # ──────────────────────────────────────────────
    # Detector 3: Drug Masking
    # ──────────────────────────────────────────────

    def _detect_drug_masking(
        self, entropy_state: dict, drug_state: dict, vitals: dict
    ) -> dict:
        """
        Fires when active drugs may be masking hemodynamic deterioration
        (e.g., vasopressors keeping BP stable while entropy drops).
        """
        drug_masking = drug_state.get("drug_masking", False)

        if not drug_masking:
            return _make_result("drug_masking")

        active_drugs = drug_state.get("active_drugs", [])
        drug_names = [d.get("drug_name", "Unknown") for d in active_drugs]

        ces = entropy_state.get("ces_adjusted", 1.0)
        sampen_bp = entropy_state.get("sampen_bp_sys", 1.0)

        # Check for concerning pattern: drugs active + entropy declining
        factors = [f"drug: {name}" for name in drug_names]

        # Check BP stability (drugs keeping it stable)
        bp_sys = vitals.get("bp_sys", 120)
        bp_stable = 90 <= bp_sys <= 160 if bp_sys else True
        factors.append(f"bp_stable: {str(bp_stable).lower()}")

        # Check if BP entropy is declining despite stable values
        bp_entropy_declining = sampen_bp < 0.50
        factors.append(f"bp_entropy_declining: {str(bp_entropy_declining).lower()}")

        severity = "WARNING"
        if ces < 0.30 and bp_entropy_declining:
            severity = "CRITICAL"

        primary_drug = drug_names[0] if drug_names else "medication"
        return _make_result(
            "drug_masking",
            active=True,
            severity=severity,
            message=f"{primary_drug} may be masking hemodynamic deterioration",
            contributing_factors=factors,
            recommended_action="Evaluate vasopressor dependency",
        )

    # ──────────────────────────────────────────────
    # Detector 4: Respiratory Risk
    # ──────────────────────────────────────────────

    def _detect_respiratory_risk(
        self, entropy_state: dict, vitals: dict, ml_predictions: dict
    ) -> dict:
        """
        Fires when respiratory-specific deterioration pattern is detected:
        - RR entropy dropping
        - SpO2 entropy dropping
        - RR values trending up (compensatory tachypnea)
        """
        sampen_rr = entropy_state.get("sampen_rr", 1.0)
        sampen_spo2 = entropy_state.get("sampen_spo2", 1.0)
        warmup = entropy_state.get("warmup_complete", False)

        if not warmup:
            return _make_result("respiratory_risk")

        factors = []
        risk_score = 0

        # Low RR entropy
        if sampen_rr < 0.40:
            risk_score += 2
            factors.append(f"sampen_rr: {sampen_rr:.2f}")
        elif sampen_rr < 0.60:
            risk_score += 1
            factors.append(f"sampen_rr: {sampen_rr:.2f}")

        # Low SpO2 entropy
        if sampen_spo2 < 0.40:
            risk_score += 2
            factors.append(f"sampen_spo2: {sampen_spo2:.2f}")
        elif sampen_spo2 < 0.60:
            risk_score += 1
            factors.append(f"sampen_spo2: {sampen_spo2:.2f}")

        # SpO2 value dropping
        spo2 = vitals.get("spo2")
        if spo2 is not None and spo2 < 94:
            risk_score += 1
            factors.append(f"spo2_value: {spo2}")

        # Elevated RR (compensatory)
        rr = vitals.get("rr")
        if rr is not None and rr > 22:
            risk_score += 1
            factors.append(f"rr_value: {rr}")

        # ML syndrome suggests respiratory
        syndrome = ml_predictions.get("syndrome")
        if syndrome and syndrome.get("primary_syndrome") == "Respiratory Failure":
            risk_score += 2
            factors.append(f"ml_syndrome: Respiratory Failure")

        if risk_score >= 3:
            severity = "CRITICAL" if risk_score >= 5 else "WARNING"
            return _make_result(
                "respiratory_risk",
                active=True,
                severity=severity,
                message="Respiratory deterioration pattern detected",
                contributing_factors=factors,
                recommended_action="Assess oxygenation and ventilation. Consider ABG.",
            )

        return _make_result("respiratory_risk")

    # ──────────────────────────────────────────────
    # Detector 5: Hemodynamic Instability
    # ──────────────────────────────────────────────

    def _detect_hemodynamic(
        self, entropy_state: dict, vitals: dict, ml_predictions: dict
    ) -> dict:
        """
        Fires when hemodynamic instability pattern is detected:
        - BP entropy declining
        - Shock index elevated (HR/BP_sys > 0.7)
        - HR entropy declining
        """
        sampen_bp = entropy_state.get("sampen_bp_sys", 1.0)
        sampen_hr = entropy_state.get("sampen_hr", 1.0)
        warmup = entropy_state.get("warmup_complete", False)

        if not warmup:
            return _make_result("hemodynamic")

        factors = []
        risk_score = 0

        # BP entropy
        if sampen_bp < 0.40:
            risk_score += 2
            factors.append(f"sampen_bp_sys: {sampen_bp:.2f}")
        elif sampen_bp < 0.60:
            risk_score += 1
            factors.append(f"sampen_bp_sys: {sampen_bp:.2f}")

        # HR entropy
        if sampen_hr < 0.40:
            risk_score += 1
            factors.append(f"sampen_hr: {sampen_hr:.2f}")

        # Shock index
        hr = vitals.get("hr")
        bp_sys = vitals.get("bp_sys")
        if hr is not None and bp_sys is not None and bp_sys > 0:
            shock_index = hr / bp_sys
            if shock_index > 0.9:
                risk_score += 2
                factors.append(f"shock_index: {shock_index:.2f}")
            elif shock_index > 0.7:
                risk_score += 1
                factors.append(f"shock_index: {shock_index:.2f}")

        # Low MAP
        bp_dia = vitals.get("bp_dia")
        if bp_sys is not None and bp_dia is not None:
            map_val = bp_dia + (bp_sys - bp_dia) / 3
            if map_val < 65:
                risk_score += 2
                factors.append(f"MAP: {map_val:.0f}")

        # ML syndrome suggests hemodynamic
        syndrome = ml_predictions.get("syndrome")
        if syndrome and syndrome.get("primary_syndrome") == "Hemodynamic Instability":
            risk_score += 2
            factors.append("ml_syndrome: Hemodynamic Instability")

        if risk_score >= 3:
            severity = "CRITICAL" if risk_score >= 5 else "WARNING" if risk_score >= 3 else "WATCH"
            return _make_result(
                "hemodynamic",
                active=True,
                severity=severity,
                message="Hemodynamic instability pattern detected",
                contributing_factors=factors,
                recommended_action="Review hemodynamic parameters",
            )
        elif risk_score >= 2:
            return _make_result(
                "hemodynamic",
                active=True,
                severity="WATCH",
                message="Hemodynamic instability pattern detected",
                contributing_factors=factors,
                recommended_action="Review hemodynamic parameters",
            )

        return _make_result("hemodynamic")

    # ──────────────────────────────────────────────
    # Detector 6: Alarm Suppression
    # ──────────────────────────────────────────────

    def _detect_alarm_suppression(
        self, vitals: dict, drug_state: dict
    ) -> dict:
        """
        Fires when a traditional alarm threshold is crossed but an active
        drug may explain the vital sign change (intelligent alarm suppression).
        """
        violated_vitals = []

        # Check each vital against traditional thresholds
        vital_map = {
            "hr": vitals.get("hr"),
            "bp_sys": vitals.get("bp_sys"),
            "bp_dia": vitals.get("bp_dia"),
            "rr": vitals.get("rr"),
            "spo2": vitals.get("spo2"),
            "temp": vitals.get("temp"),
        }

        for vital_name, value in vital_map.items():
            if value is None:
                continue
            thresh = TRADITIONAL_THRESHOLDS.get(vital_name)
            if thresh is None:
                continue

            low = thresh.get("low")
            high = thresh.get("high")

            if low is not None and value < low:
                violated_vitals.append((vital_name, "low", value))
            if high is not None and value > high:
                violated_vitals.append((vital_name, "high", value))

        if not violated_vitals:
            return _make_result("alarm_suppression")

        # Check if any active drug can explain the vital change
        active_drugs = drug_state.get("active_drugs", [])
        drug_masking = drug_state.get("drug_masking", False)

        if not drug_masking or not active_drugs:
            return _make_result("alarm_suppression")

        # At least one violated vital + drug masking active = alarm suppression
        factors = []
        for vital_name, direction, value in violated_vitals:
            factors.append(f"{vital_name}_{direction}: {value}")

        drug_names = [d.get("drug_name", "Unknown") for d in active_drugs]
        factors.extend([f"explaining_drug: {name}" for name in drug_names])

        # Track suppressed alarm count
        # (Person 3 may expose this counter via API)
        self.suppressed_alarm_count["total"] = (
            self.suppressed_alarm_count.get("total", 0) + 1
        )

        return _make_result(
            "alarm_suppression",
            active=True,
            severity="WATCH",
            message=(
                f"Traditional alarm suppressed — {', '.join(drug_names)} "
                f"may explain {', '.join(v[0] for v in violated_vitals)} changes"
            ),
            contributing_factors=factors,
            recommended_action="Alarm suppressed by drug context. Clinical review if persistent.",
        )

    # ──────────────────────────────────────────────
    # Detector 7: Recovery
    # ──────────────────────────────────────────────

    def _detect_recovery(
        self, entropy_state: dict, ml_predictions: dict, fusion: dict
    ) -> dict:
        """
        Fires when patient shows signs of recovery:
        - CES slope positive (entropy rising — complexity returning)
        - ML risk declining
        - Previously elevated severity now decreasing
        """
        ces_slope = entropy_state.get("ces_slope_6h", 0.0)
        ces = entropy_state.get("ces_adjusted", 1.0)
        warmup = entropy_state.get("warmup_complete", False)

        if not warmup:
            return _make_result("recovery")

        factors = []
        recovery_score = 0

        # CES slope positive (entropy recovering)
        if ces_slope > 0.001:
            recovery_score += 2
            factors.append(f"ces_slope_6h: +{ces_slope:.4f}")
        elif ces_slope > 0.0005:
            recovery_score += 1
            factors.append(f"ces_slope_6h: +{ces_slope:.4f}")

        # CES still in moderate range but improving
        if 0.40 < ces < 0.65 and ces_slope > 0:
            recovery_score += 1
            factors.append(f"ces_recovering: {ces:.2f}")

        # ML risk is low (if available)
        det = ml_predictions.get("deterioration")
        if det and isinstance(det, dict):
            risk_4h = det.get("risk_4h", 1.0)
            if risk_4h < 0.20:
                recovery_score += 1
                factors.append(f"ml_risk_4h: {risk_4h:.2f}")

        # Fusion severity not high
        severity = fusion.get("final_severity", "NONE")
        if severity in ("NONE", "WATCH"):
            recovery_score += 1

        if recovery_score >= 3:
            return _make_result(
                "recovery",
                active=True,
                severity="WATCH",
                message="Recovery trend — entropy complexity returning to normal",
                contributing_factors=factors,
                recommended_action="Continue monitoring. Consider reducing surveillance level.",
            )

        return _make_result("recovery")

    # ──────────────────────────────────────────────
    # Detector 8: Data Quality
    # ──────────────────────────────────────────────

    def _detect_data_quality(self, entropy_state: dict, vitals: dict) -> dict:
        """
        Fires when data quality issues are detected:
        - Missing vital signs
        - Physiologically implausible values
        - Entropy still in warmup
        """
        factors = []
        issues = 0

        # Check for warmup
        warmup = entropy_state.get("warmup_complete", False)
        if not warmup:
            issues += 1
            factors.append("entropy: still_calibrating")

        # Check for missing vitals
        required_vitals = ["hr", "bp_sys", "rr", "spo2"]
        for vital in required_vitals:
            if vitals.get(vital) is None:
                issues += 1
                factors.append(f"{vital}: missing")

        # Check for physiologically implausible values
        plausibility = {
            "hr": (20, 250),
            "bp_sys": (40, 300),
            "bp_dia": (20, 200),
            "rr": (2, 60),
            "spo2": (50, 100),
            "temp": (30.0, 45.0),
        }

        for vital, (low, high) in plausibility.items():
            val = vitals.get(vital)
            if val is not None and (val < low or val > high):
                issues += 1
                factors.append(f"{vital}_implausible: {val}")

        if issues > 0:
            severity = "WARNING" if issues >= 3 else "WATCH"
            return _make_result(
                "data_quality",
                active=True,
                severity=severity,
                message=f"Data quality concern — {issues} issue(s) detected",
                contributing_factors=factors,
                recommended_action="Verify sensor connections and data integrity",
            )

        return _make_result("data_quality")

    # ──────────────────────────────────────────────
    # Utility methods
    # ──────────────────────────────────────────────

    def _are_vitals_normal(self, vitals: dict) -> bool:
        """
        Check whether all vital values are within normal clinical range.
        Used by the silent decline detector.
        """
        normal_ranges = {
            "hr": (55, 100),
            "bp_sys": (95, 145),
            "bp_dia": (55, 90),
            "rr": (10, 22),
            "spo2": (93, 100),
            "temp": (36.0, 38.0),
        }

        for vital, (low, high) in normal_ranges.items():
            val = vitals.get(vital)
            if val is not None and (val < low or val > high):
                return False

        return True

    def get_suppressed_alarm_count(self) -> int:
        """Return the total number of suppressed alarms."""
        return self.suppressed_alarm_count.get("total", 0)
