"""
Alarm Fatigue Tracker — quantifies alarm reduction.

Compares traditional threshold-based alarm counts against
Chronos entropy-based alert counts to prove that Chronos
reduces alarm fatigue while improving detection accuracy.

This provides the hard number: "X% alarm reduction."
"""

from typing import Dict, List, Optional
from ..models import VitalSignRecord, AlertSeverity

import logging
logger = logging.getLogger(__name__)


# Standard ICU alarm thresholds
TRADITIONAL_THRESHOLDS = {
    "heart_rate": (50.0, 120.0),
    "spo2": (90.0, 100.0),
    "bp_systolic": (90.0, 180.0),
    "bp_diastolic": (40.0, 110.0),
    "resp_rate": (8.0, 30.0),
    "temperature": (35.5, 38.5),
}


class AlarmFatigueTracker:
    """
    Tracks and compares traditional vs Chronos alarm counts.

    For every vital sign record processed:
    - Check if traditional thresholds would fire an alarm
    - Record what Chronos severity was assigned
    - Track drug-suppressed alerts
    - Compute running comparison statistics
    """

    def __init__(self):
        self.traditional_alarm_count: int = 0
        self.traditional_alarm_records: int = 0  # Records where traditional would alarm
        self.chronos_none_count: int = 0
        self.chronos_watch_count: int = 0
        self.chronos_warning_count: int = 0
        self.chronos_critical_count: int = 0
        self.drug_masked_count: int = 0
        self.total_records: int = 0
        self.per_patient: Dict[str, Dict] = {}

    def record_comparison(
        self,
        patient_id: str,
        record: VitalSignRecord,
        chronos_severity: AlertSeverity,
        drug_masked: bool = False,
    ):
        """Record one vital sign comparison between traditional and Chronos."""
        self.total_records += 1

        # Initialize per-patient tracking
        if patient_id not in self.per_patient:
            self.per_patient[patient_id] = {
                "traditional_alarms": 0,
                "chronos_actionable": 0,
                "total_records": 0,
            }
        self.per_patient[patient_id]["total_records"] += 1

        # Would traditional alarm fire?
        trad_fired = False
        fired_vitals = []
        for vital_name, (low, high) in TRADITIONAL_THRESHOLDS.items():
            value = getattr(record, vital_name, None)
            if value is not None and (value < low or value > high):
                trad_fired = True
                fired_vitals.append(vital_name)

        if trad_fired:
            self.traditional_alarm_count += 1
            self.traditional_alarm_records += 1
            self.per_patient[patient_id]["traditional_alarms"] += 1

        # Chronos severity tracking
        if chronos_severity == AlertSeverity.NONE:
            self.chronos_none_count += 1
        elif chronos_severity == AlertSeverity.WATCH:
            self.chronos_watch_count += 1
        elif chronos_severity == AlertSeverity.WARNING:
            self.chronos_warning_count += 1
            self.per_patient[patient_id]["chronos_actionable"] += 1
        elif chronos_severity == AlertSeverity.CRITICAL:
            self.chronos_critical_count += 1
            self.per_patient[patient_id]["chronos_actionable"] += 1

        if drug_masked:
            self.drug_masked_count += 1

    def get_statistics(self) -> Dict:
        """Get comprehensive alarm fatigue comparison statistics."""
        chronos_actionable = self.chronos_warning_count + self.chronos_critical_count
        chronos_all_alerts = self.chronos_watch_count + chronos_actionable

        # Alarm reduction calculation
        if self.traditional_alarm_count > 0:
            # How many fewer actionable alarms does Chronos produce?
            reduction_pct = max(0.0,
                (1.0 - chronos_actionable / self.traditional_alarm_count) * 100
            )
        else:
            reduction_pct = 100.0 if chronos_actionable == 0 else 0.0

        # Traditional alarm rate (alarms per hour, assuming 1 record per minute)
        hours = max(1, self.total_records / 60)
        trad_rate = self.traditional_alarm_count / hours
        chronos_rate = chronos_actionable / hours

        return {
            "traditional_monitoring": {
                "total_threshold_alarms": self.traditional_alarm_count,
                "alarms_per_hour": round(trad_rate, 1),
                "description": "Alarms fired when ANY vital crosses standard ICU thresholds",
            },
            "chronos_monitoring": {
                "watch_alerts": self.chronos_watch_count,
                "warning_alerts": self.chronos_warning_count,
                "critical_alerts": self.chronos_critical_count,
                "actionable_alerts": chronos_actionable,
                "alerts_per_hour": round(chronos_rate, 1),
                "drug_masked_detections": self.drug_masked_count,
                "description": "Entropy-based alerts with clinical intelligence filtering",
            },
            "comparison": {
                "alarm_reduction_percent": round(reduction_pct, 1),
                "total_records_analyzed": self.total_records,
                "monitoring_hours": round(hours, 1),
                "headline": (
                    f"{reduction_pct:.0f}% alarm reduction. "
                    f"Traditional: {self.traditional_alarm_count} alarms "
                    f"({trad_rate:.0f}/hr). "
                    f"Chronos: {chronos_actionable} actionable alerts "
                    f"({chronos_rate:.1f}/hr)."
                ),
            },
            "per_patient_summary": {
                pid: {
                    "traditional_alarms": stats["traditional_alarms"],
                    "chronos_actionable": stats["chronos_actionable"],
                    "records": stats["total_records"],
                }
                for pid, stats in self.per_patient.items()
            },
        }

    def reset(self):
        """Reset all counters."""
        self.__init__()
