"""
Digital Twin Data Mapper — Maps patient analytics to body regions.

Translates entropy, vital signs, correlations, and alerts into
body-region-specific danger levels for 3D human model.

Each body region gets:
  - danger_level: float 0-1 (0=healthy, 1=critical)
  - color: hex color string
  - status: text label
  - metrics: dict of associated measurements
  - alerts: list of relevant warnings
"""

from typing import Dict, List, Optional, Any
from ..models import PatientState, AlertSeverity, TrendDirection


# Color gradient from healthy to critical
COLORS = {
    "healthy": "#00E676",
    "watch": "#FFEB3B",
    "warning": "#FF9800",
    "critical": "#F44336",
    "inactive": "#616161",
}


def _danger_color(level: float) -> str:
    """Map danger level (0-1) to hex color."""
    if level < 0.25:
        return COLORS["healthy"]
    elif level < 0.50:
        return COLORS["watch"]
    elif level < 0.75:
        return COLORS["warning"]
    else:
        return COLORS["critical"]


def _danger_status(level: float) -> str:
    """Map danger level (0-1) to status label."""
    if level < 0.25:
        return "Normal"
    elif level < 0.50:
        return "Watch"
    elif level < 0.75:
        return "Warning"
    else:
        return "Critical"


class DigitalTwinMapper:
    """
    Maps patient state data to body regions for 3D visualization.

    Body regions:
      - brain: temperature, consciousness indicators
      - heart: heart rate, cardiac entropy, cardiac coupling
      - lungs: SpO2, respiratory rate, respiratory entropy
      - vessels: blood pressure, vascular coupling
      - autonomic: overall autonomic function (baroreflex, CES)
      - abdomen: temperature trends, systemic indicators
      - overall: composite danger assessment
    """

    def map_patient(
        self,
        state: PatientState,
        cross_correlations: Optional[Dict] = None,
        decoupling_summary: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Map a patient's complete state to body region data.

        Parameters
        ----------
        state : PatientState
            Full patient state with all analytics
        cross_correlations : dict, optional
            Output from CrossVitalAnalyzer.compute_correlations()
        decoupling_summary : dict, optional
            Output from CrossVitalAnalyzer.get_decoupling_summary()

        Returns
        -------
        Dict with body regions, each containing danger_level,
        color, status, metrics, and alerts.
        """
        if state.calibrating:
            return self._calibrating_response(state)

        regions = {}

        regions["brain"] = self._map_brain(state, cross_correlations)
        regions["heart"] = self._map_heart(state, cross_correlations)
        regions["lungs"] = self._map_lungs(state, cross_correlations)
        regions["vessels"] = self._map_vessels(state, cross_correlations)
        regions["autonomic"] = self._map_autonomic(
            state, cross_correlations, decoupling_summary
        )
        regions["abdomen"] = self._map_abdomen(state)

        # Overall danger is max of all regions
        max_danger = max(
            r["danger_level"] for r in regions.values()
        )
        regions["overall"] = {
            "danger_level": round(max_danger, 3),
            "color": _danger_color(max_danger),
            "status": _danger_status(max_danger),
            "composite_entropy": round(state.composite_entropy, 3),
            "alert_severity": state.alert.severity.value,
        }

        return {
            "patient_id": state.patient_id,
            "timestamp": state.timestamp.isoformat(),
            "calibrating": False,
            "regions": regions,
            "active_drugs": [
                {
                    "name": d.drug_name,
                    "drug_class": d.drug_class,
                }
                for d in state.active_drugs
            ],
            "drug_masked": state.alert.drug_masked,
        }

    def _map_brain(
        self, state: PatientState, corr: Optional[Dict]
    ) -> Dict:
        """Brain region: temperature + consciousness indicators."""
        temp_detail = state.vitals.temperature
        temp_val = temp_detail.value if temp_detail else None
        temp_entropy = temp_detail.sampen_normalized if temp_detail else None

        danger = 0.0
        alerts = []

        if temp_val is not None:
            if temp_val > 39.0 or temp_val < 35.5:
                danger = max(danger, 0.8)
                alerts.append(
                    f"Temperature abnormal: {temp_val:.1f}C"
                )
            elif temp_val > 38.0 or temp_val < 36.0:
                danger = max(danger, 0.4)

        if temp_entropy is not None and temp_entropy < 0.3:
            danger = max(danger, 0.5)
            alerts.append("Temperature entropy declining")

        return {
            "danger_level": round(danger, 3),
            "color": _danger_color(danger),
            "status": _danger_status(danger),
            "metrics": {
                "temperature": temp_val,
                "temperature_entropy": (
                    round(temp_entropy, 3) if temp_entropy else None
                ),
            },
            "alerts": alerts,
        }

    def _map_heart(
        self, state: PatientState, corr: Optional[Dict]
    ) -> Dict:
        """Heart region: heart rate + cardiac entropy + coupling."""
        hr_detail = state.vitals.heart_rate
        hr_val = hr_detail.value if hr_detail else None
        hr_entropy = hr_detail.sampen_normalized if hr_detail else None
        hr_trend = hr_detail.trend if hr_detail else None

        danger = 0.0
        alerts = []

        # Heart rate value check
        if hr_val is not None:
            if hr_val > 130 or hr_val < 45:
                danger = max(danger, 0.9)
                alerts.append(f"Heart rate critical: {hr_val:.0f} bpm")
            elif hr_val > 110 or hr_val < 55:
                danger = max(danger, 0.5)

        # Heart rate entropy check
        if hr_entropy is not None:
            if hr_entropy < 0.2:
                danger = max(danger, 0.8)
                alerts.append(
                    f"Heart rate entropy critically low: "
                    f"{hr_entropy:.2f}"
                )
            elif hr_entropy < 0.4:
                danger = max(danger, 0.5)
                alerts.append("Heart rate variability declining")

        # Cardiac coupling check
        if corr:
            cp = corr.get("heart_rate__resp_rate", {})
            if cp.get("decoupled", False):
                danger = max(danger, 0.6)
                alerts.append("Cardiopulmonary decoupling detected")

            co = corr.get("heart_rate__spo2", {})
            if co.get("decoupled", False):
                danger = max(danger, 0.6)
                alerts.append(
                    "Cardiac-oxygenation decoupling detected"
                )

        # Trend bonus
        if hr_trend == TrendDirection.FALLING and hr_entropy is not None:
            if hr_entropy < 0.4:
                danger = min(1.0, danger + 0.1)

        return {
            "danger_level": round(min(1.0, danger), 3),
            "color": _danger_color(danger),
            "status": _danger_status(danger),
            "metrics": {
                "heart_rate": hr_val,
                "heart_rate_entropy": (
                    round(hr_entropy, 3) if hr_entropy else None
                ),
                "trend": hr_trend.value if hr_trend else "stable",
            },
            "alerts": alerts,
        }

    def _map_lungs(
        self, state: PatientState, corr: Optional[Dict]
    ) -> Dict:
        """Lungs region: SpO2 + respiratory rate + coupling."""
        spo2_detail = state.vitals.spo2
        rr_detail = state.vitals.resp_rate

        spo2_val = spo2_detail.value if spo2_detail else None
        spo2_entropy = (
            spo2_detail.sampen_normalized if spo2_detail else None
        )
        rr_val = rr_detail.value if rr_detail else None
        rr_entropy = (
            rr_detail.sampen_normalized if rr_detail else None
        )

        danger = 0.0
        alerts = []

        # SpO2 check
        if spo2_val is not None:
            if spo2_val < 88:
                danger = max(danger, 0.95)
                alerts.append(f"SpO2 critical: {spo2_val:.0f}%")
            elif spo2_val < 92:
                danger = max(danger, 0.7)
                alerts.append(f"SpO2 low: {spo2_val:.0f}%")
            elif spo2_val < 95:
                danger = max(danger, 0.4)

        # Respiratory rate check
        if rr_val is not None:
            if rr_val > 28 or rr_val < 8:
                danger = max(danger, 0.8)
                alerts.append(
                    f"Respiratory rate abnormal: {rr_val:.0f}/min"
                )
            elif rr_val > 24 or rr_val < 10:
                danger = max(danger, 0.5)

        # Respiratory entropy check
        if rr_entropy is not None and rr_entropy < 0.2:
            danger = max(danger, 0.7)
            alerts.append(
                "Respiratory pattern entropy critically low"
            )
        elif rr_entropy is not None and rr_entropy < 0.4:
            danger = max(danger, 0.4)

        # Respiratory drive coupling
        if corr:
            rd = corr.get("resp_rate__spo2", {})
            if rd.get("decoupled", False):
                danger = max(danger, 0.7)
                alerts.append(
                    "Respiratory drive decoupling: compensation "
                    "failure"
                )

        return {
            "danger_level": round(min(1.0, danger), 3),
            "color": _danger_color(danger),
            "status": _danger_status(danger),
            "metrics": {
                "spo2": spo2_val,
                "spo2_entropy": (
                    round(spo2_entropy, 3) if spo2_entropy else None
                ),
                "resp_rate": rr_val,
                "resp_rate_entropy": (
                    round(rr_entropy, 3) if rr_entropy else None
                ),
            },
            "alerts": alerts,
        }

    def _map_vessels(
        self, state: PatientState, corr: Optional[Dict]
    ) -> Dict:
        """Vessels region: blood pressure + vascular coupling."""
        sys_detail = state.vitals.bp_systolic
        dia_detail = state.vitals.bp_diastolic

        sys_val = sys_detail.value if sys_detail else None
        sys_entropy = (
            sys_detail.sampen_normalized if sys_detail else None
        )
        dia_val = dia_detail.value if dia_detail else None

        danger = 0.0
        alerts = []

        # Blood pressure check
        if sys_val is not None:
            if sys_val < 85 or sys_val > 190:
                danger = max(danger, 0.9)
                alerts.append(
                    f"Blood pressure critical: {sys_val:.0f} mmHg"
                )
            elif sys_val < 95 or sys_val > 170:
                danger = max(danger, 0.5)

        # Pulse pressure check
        if sys_val is not None and dia_val is not None:
            pp = sys_val - dia_val
            if pp > 60 or pp < 25:
                danger = max(danger, 0.5)
                alerts.append(f"Pulse pressure abnormal: {pp:.0f}")

        # BP entropy check
        if sys_entropy is not None and sys_entropy < 0.2:
            danger = max(danger, 0.7)
            alerts.append(
                "Blood pressure entropy critically low"
            )

        # Vascular coupling check
        if corr:
            vc = corr.get("bp_systolic__bp_diastolic", {})
            if vc.get("decoupled", False):
                danger = max(danger, 0.8)
                alerts.append(
                    "Vascular tone decoupling: suggests "
                    "vasodilatory shock"
                )

        return {
            "danger_level": round(min(1.0, danger), 3),
            "color": _danger_color(danger),
            "status": _danger_status(danger),
            "metrics": {
                "bp_systolic": sys_val,
                "bp_diastolic": dia_val,
                "bp_entropy": (
                    round(sys_entropy, 3) if sys_entropy else None
                ),
                "pulse_pressure": (
                    round(sys_val - dia_val, 0)
                    if sys_val and dia_val
                    else None
                ),
            },
            "alerts": alerts,
        }

    def _map_autonomic(
        self,
        state: PatientState,
        corr: Optional[Dict],
        decoupling: Optional[Dict],
    ) -> Dict:
        """Autonomic region: overall autonomic function."""
        danger = 0.0
        alerts = []

        # CES-based autonomic assessment
        ces = state.composite_entropy
        if ces < 0.2:
            danger = max(danger, 0.9)
            alerts.append(
                f"Autonomic function severely compromised "
                f"(CES: {ces:.2f})"
            )
        elif ces < 0.4:
            danger = max(danger, 0.6)
            alerts.append(
                f"Autonomic function declining (CES: {ces:.2f})"
            )
        elif ces < 0.6:
            danger = max(danger, 0.3)

        # Baroreflex coupling
        if corr:
            br = corr.get("heart_rate__bp_systolic", {})
            if br.get("decoupled", False):
                danger = max(danger, 0.7)
                alerts.append(
                    "Baroreflex arc disrupted: autonomic "
                    "failure likely"
                )

        # Multi-system decoupling
        if decoupling:
            n_dec = decoupling.get("decoupled_count", 0)
            total = decoupling.get("total_pairs", 5)
            if n_dec >= 3:
                danger = max(danger, 0.95)
                alerts.append(
                    f"Multi-organ decoupling: {n_dec}/{total} "
                    f"systems disconnected"
                )
            elif n_dec >= 2:
                danger = max(danger, 0.7)

        return {
            "danger_level": round(min(1.0, danger), 3),
            "color": _danger_color(danger),
            "status": _danger_status(danger),
            "metrics": {
                "composite_entropy": round(ces, 3),
                "decoupled_systems": (
                    decoupling.get("decoupled_count", 0)
                    if decoupling
                    else 0
                ),
                "total_monitored_couplings": (
                    decoupling.get("total_pairs", 0)
                    if decoupling
                    else 0
                ),
            },
            "alerts": alerts,
        }

    def _map_abdomen(self, state: PatientState) -> Dict:
        """Abdomen region: systemic indicators."""
        temp_detail = state.vitals.temperature
        temp_val = temp_detail.value if temp_detail else None

        danger = 0.0
        alerts = []

        # Systemic infection indicators
        if temp_val is not None and temp_val > 38.3:
            danger = max(danger, 0.5)
            alerts.append(
                f"Febrile: {temp_val:.1f}C "
                f"(possible systemic infection)"
            )

        # Low entropy across multiple vitals = systemic issue
        low_entropy_count = 0
        for name in [
            "heart_rate", "spo2", "bp_systolic",
            "resp_rate",
        ]:
            detail = getattr(state.vitals, name, None)
            if (
                detail
                and detail.sampen_normalized is not None
                and detail.sampen_normalized < 0.3
            ):
                low_entropy_count += 1

        if low_entropy_count >= 3:
            danger = max(danger, 0.8)
            alerts.append(
                f"{low_entropy_count}/4 vital signs show low "
                f"entropy: systemic deterioration"
            )
        elif low_entropy_count >= 2:
            danger = max(danger, 0.5)

        return {
            "danger_level": round(min(1.0, danger), 3),
            "color": _danger_color(danger),
            "status": _danger_status(danger),
            "metrics": {
                "temperature": temp_val,
                "low_entropy_vital_count": low_entropy_count,
            },
            "alerts": alerts,
        }

    def _calibrating_response(self, state: PatientState) -> Dict:
        """Return a neutral response while calibrating."""
        regions = {}
        for region_name in [
            "brain", "heart", "lungs", "vessels",
            "autonomic", "abdomen",
        ]:
            regions[region_name] = {
                "danger_level": 0.0,
                "color": COLORS["inactive"],
                "status": "Calibrating",
                "metrics": {},
                "alerts": [],
            }
        regions["overall"] = {
            "danger_level": 0.0,
            "color": COLORS["inactive"],
            "status": "Calibrating",
            "composite_entropy": 1.0,
            "alert_severity": "NONE",
        }
        return {
            "patient_id": state.patient_id,
            "timestamp": state.timestamp.isoformat(),
            "calibrating": True,
            "regions": regions,
            "active_drugs": [],
            "drug_masked": False,
        }
