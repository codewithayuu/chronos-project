"""
Chart Data Formatter — Pre-formats data for frontend charts.

Returns arrays of objects ready for Recharts / Chart.js consumption.
No transformation needed on the frontend.
"""

from typing import Dict, Any, List, Optional
from ..models import PatientState, AlertSeverity


class ChartDataFormatter:
    """Formats patient and system data for chart rendering."""

    def patient_charts(
        self,
        state: PatientState,
        history: List[PatientState],
        cross_correlations: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate all chart data for a single patient.

        Returns dict with chart-ready arrays for:
          - entropy_pie: CES breakdown by vital sign
          - vital_bars: current vital values as bar chart
          - entropy_trend: CES over time
          - vital_entropy_comparison: value vs entropy per vital
          - correlation_matrix: heatmap data
          - severity_timeline: alert levels over time
        """
        charts = {}

        charts["entropy_pie"] = self._entropy_pie(state)
        charts["vital_bars"] = self._vital_bars(state)
        charts["entropy_trend"] = self._entropy_trend(history)
        charts["vital_entropy_comparison"] = (
            self._vital_entropy_comparison(state)
        )
        charts["correlation_matrix"] = (
            self._correlation_matrix(cross_correlations)
        )
        charts["severity_timeline"] = (
            self._severity_timeline(history)
        )
        charts["clinical_score_gauge"] = (
            self._clinical_score_gauge(state)
        )

        return charts

    def system_dashboard(
        self,
        all_states: Dict[str, PatientState],
        alarm_stats: Dict,
        validation_report: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate dashboard-level chart data.

        Returns:
          - patient_severity_distribution: pie chart
          - alarm_comparison: bar chart traditional vs chronos
          - patient_entropy_ranking: sorted bar chart
          - validation_metrics: if available
        """
        charts = {}

        charts["patient_severity_distribution"] = (
            self._severity_distribution(all_states)
        )
        charts["alarm_comparison"] = (
            self._alarm_comparison(alarm_stats)
        )
        charts["patient_entropy_ranking"] = (
            self._entropy_ranking(all_states)
        )

        if validation_report and "summary" in validation_report:
            charts["validation_metrics"] = (
                self._validation_charts(validation_report)
            )

        return charts

    def _entropy_pie(self, state: PatientState) -> List[Dict]:
        """CES breakdown by vital sign — for pie/donut chart."""
        weights = {
            "Heart Rate": 0.25,
            "SpO2": 0.15,
            "Systolic BP": 0.20,
            "Diastolic BP": 0.10,
            "Resp Rate": 0.20,
            "Temperature": 0.10,
        }
        attrs = {
            "Heart Rate": "heart_rate",
            "SpO2": "spo2",
            "Systolic BP": "bp_systolic",
            "Diastolic BP": "bp_diastolic",
            "Resp Rate": "resp_rate",
            "Temperature": "temperature",
        }

        data = []
        for label, attr in attrs.items():
            detail = getattr(state.vitals, attr, None)
            entropy = (
                detail.sampen_normalized
                if detail and detail.sampen_normalized is not None
                else 0.5
            )
            weight = weights[label]
            contribution = entropy * weight

            data.append({
                "name": label,
                "entropy": round(entropy, 3),
                "weight": weight,
                "contribution": round(contribution, 4),
                "color": self._entropy_color(entropy),
            })

        return data

    def _vital_bars(self, state: PatientState) -> List[Dict]:
        """Current vital sign values for bar chart."""
        configs = [
            ("Heart Rate", "heart_rate", "bpm", 50, 120),
            ("SpO2", "spo2", "%", 90, 100),
            ("Systolic BP", "bp_systolic", "mmHg", 90, 180),
            ("Diastolic BP", "bp_diastolic", "mmHg", 40, 110),
            ("Resp Rate", "resp_rate", "/min", 8, 30),
            ("Temp", "temperature", "C", 35.5, 38.5),
        ]

        data = []
        for label, attr, unit, low, high in configs:
            detail = getattr(state.vitals, attr, None)
            value = detail.value if detail else None

            in_range = True
            if value is not None:
                in_range = low <= value <= high

            data.append({
                "name": label,
                "value": round(value, 1) if value else None,
                "unit": unit,
                "normal_low": low,
                "normal_high": high,
                "in_range": in_range,
                "color": (
                    "#00E676" if in_range else "#F44336"
                ),
            })

        return data

    def _entropy_trend(
        self, history: List[PatientState]
    ) -> List[Dict]:
        """CES over time for line chart."""
        data = []
        for i, state in enumerate(history):
            if not state.calibrating:
                data.append({
                    "minute": i,
                    "ces": round(state.composite_entropy, 4),
                    "severity": state.alert.severity.value,
                    "color": self._severity_color(
                        state.alert.severity
                    ),
                })
        return data

    def _vital_entropy_comparison(
        self, state: PatientState
    ) -> List[Dict]:
        """Side-by-side value vs entropy per vital."""
        vitals = [
            ("Heart Rate", "heart_rate"),
            ("SpO2", "spo2"),
            ("Systolic BP", "bp_systolic"),
            ("Resp Rate", "resp_rate"),
            ("Temperature", "temperature"),
        ]

        data = []
        for label, attr in vitals:
            detail = getattr(state.vitals, attr, None)
            if detail is None:
                continue

            value = detail.value
            entropy = detail.sampen_normalized

            data.append({
                "name": label,
                "value": (
                    round(value, 1) if value is not None else None
                ),
                "entropy": (
                    round(entropy, 3)
                    if entropy is not None
                    else None
                ),
                "trend": (
                    detail.trend.value if detail.trend else "stable"
                ),
                "entropy_color": (
                    self._entropy_color(entropy)
                    if entropy is not None
                    else "#616161"
                ),
            })

        return data

    def _correlation_matrix(
        self, correlations: Optional[Dict]
    ) -> List[Dict]:
        """Correlation matrix data for heatmap."""
        if not correlations:
            return []

        data = []
        for key, info in correlations.items():
            if not info.get("data_available", False):
                continue

            v1, v2 = key.split("__")
            data.append({
                "pair": info.get("pair_name", key),
                "vital1": v1.replace("_", " ").title(),
                "vital2": v2.replace("_", " ").title(),
                "current": info.get("current"),
                "expected": info.get("expected"),
                "deviation": info.get("deviation"),
                "decoupled": info.get("decoupled", False),
                "color": (
                    "#F44336"
                    if info.get("decoupled")
                    else "#00E676"
                ),
            })

        return data

    def _severity_timeline(
        self, history: List[PatientState]
    ) -> List[Dict]:
        """Severity levels over time."""
        severity_values = {
            AlertSeverity.NONE: 0,
            AlertSeverity.WATCH: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.CRITICAL: 3,
        }

        data = []
        for i, state in enumerate(history):
            if not state.calibrating:
                sev = state.alert.severity
                data.append({
                    "minute": i,
                    "severity_value": severity_values.get(sev, 0),
                    "severity_label": sev.value,
                    "color": self._severity_color(sev),
                })
        return data

    def _clinical_score_gauge(
        self, state: PatientState
    ) -> Dict:
        """Clinical scores for gauge charts."""
        if state.clinical_scores is None:
            return {
                "news2": {"score": 0, "max": 20, "risk": "N/A"},
                "qsofa": {"score": 0, "max": 3, "risk": "N/A"},
            }

        news2 = state.clinical_scores.get("news2", {})
        qsofa = state.clinical_scores.get("qsofa", {})

        return {
            "news2": {
                "score": news2.get("score", 0),
                "max": 20,
                "risk": news2.get("risk_level", "N/A"),
                "color": self._risk_color(
                    news2.get("risk_level", "")
                ),
            },
            "qsofa": {
                "score": qsofa.get("score", 0),
                "max": 3,
                "risk": qsofa.get("risk_level", "N/A"),
                "color": self._risk_color(
                    qsofa.get("risk_level", "")
                ),
            },
        }

    def _severity_distribution(
        self, states: Dict[str, PatientState]
    ) -> List[Dict]:
        """Patient count by severity for pie chart."""
        counts = {"NONE": 0, "WATCH": 0, "WARNING": 0, "CRITICAL": 0}
        for state in states.values():
            sev = state.alert.severity.value
            counts[sev] = counts.get(sev, 0) + 1

        return [
            {
                "name": sev,
                "count": count,
                "color": self._severity_color_by_name(sev),
            }
            for sev, count in counts.items()
            if count > 0
        ]

    def _alarm_comparison(self, alarm_stats: Dict) -> List[Dict]:
        """Traditional vs Chronos alarm counts for bar chart."""
        trad = alarm_stats.get("traditional_monitoring", {})
        chronos = alarm_stats.get("chronos_monitoring", {})

        return [
            {
                "name": "Traditional Alarms",
                "count": trad.get("total_threshold_alarms", 0),
                "color": "#F44336",
            },
            {
                "name": "Chronos Watch",
                "count": chronos.get("watch_alerts", 0),
                "color": "#FFEB3B",
            },
            {
                "name": "Chronos Warning",
                "count": chronos.get("warning_alerts", 0),
                "color": "#FF9800",
            },
            {
                "name": "Chronos Critical",
                "count": chronos.get("critical_alerts", 0),
                "color": "#F44336",
            },
        ]

    def _entropy_ranking(
        self, states: Dict[str, PatientState]
    ) -> List[Dict]:
        """Patients ranked by entropy for bar chart."""
        data = []
        for pid, state in states.items():
            data.append({
                "patient_id": pid,
                "entropy": round(state.composite_entropy, 3),
                "severity": state.alert.severity.value,
                "color": self._severity_color(
                    state.alert.severity
                ),
            })

        data.sort(key=lambda x: x["entropy"])
        return data

    def _validation_charts(
        self, report: Dict
    ) -> Dict:
        """Validation metrics for display."""
        summary = report.get("summary", {})
        return {
            "sensitivity_comparison": [
                {
                    "name": "Chronos",
                    "value": summary.get(
                        "chronos_sensitivity", 0
                    ),
                    "color": "#00E676",
                },
                {
                    "name": "Traditional",
                    "value": summary.get(
                        "traditional_sensitivity", 0
                    ),
                    "color": "#F44336",
                },
            ],
            "lead_time_comparison": [
                {
                    "name": "Chronos (min)",
                    "value": summary.get(
                        "mean_chronos_lead_minutes", 0
                    ),
                    "color": "#00E676",
                },
                {
                    "name": "Traditional (min)",
                    "value": summary.get(
                        "mean_traditional_lead_minutes", 0
                    ),
                    "color": "#F44336",
                },
            ],
        }

    @staticmethod
    def _entropy_color(entropy: float) -> str:
        if entropy >= 0.6:
            return "#00E676"
        elif entropy >= 0.4:
            return "#FFEB3B"
        elif entropy >= 0.2:
            return "#FF9800"
        else:
            return "#F44336"

    @staticmethod
    def _severity_color(severity: AlertSeverity) -> str:
        return {
            AlertSeverity.NONE: "#00E676",
            AlertSeverity.WATCH: "#FFEB3B",
            AlertSeverity.WARNING: "#FF9800",
            AlertSeverity.CRITICAL: "#F44336",
        }.get(severity, "#616161")

    @staticmethod
    def _severity_color_by_name(name: str) -> str:
        return {
            "NONE": "#00E676",
            "WATCH": "#FFEB3B",
            "WARNING": "#FF9800",
            "CRITICAL": "#F44336",
        }.get(name, "#616161")

    @staticmethod
    def _risk_color(risk: str) -> str:
        return {
            "None": "#00E676",
            "Low": "#00E676",
            "Medium": "#FF9800",
            "High": "#F44336",
        }.get(risk, "#616161")
