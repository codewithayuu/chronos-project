
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque

from ..config import AppConfig, load_config
from ..models import (
    VitalSignRecord,
    PatientState,
    PatientSummary,
    DrugEffect,
    AlertSeverity,
)
from ..entropy.engine import EntropyEngine
from ..drugs.database import DrugDatabase
from ..drugs.filter import DrugFilter
from ..evidence.engine import EvidenceEngine
from ..analytics.clinical_scores import ClinicalScores
from ..analytics.alarm_fatigue import AlarmFatigueTracker
from ..analytics.cross_correlation import CrossVitalAnalyzer


# Traditional ICU alarm thresholds
_TRADITIONAL_THRESHOLDS = {
    "heart_rate": (50.0, 120.0),
    "spo2": (90.0, 100.0),
    "bp_systolic": (90.0, 180.0),
    "bp_diastolic": (40.0, 110.0),
    "resp_rate": (8.0, 30.0),
    "temperature": (35.5, 38.5),
}


class StoredAlert:
    """An alert record with lifecycle tracking."""

    def __init__(
        self,
        patient_id: str,
        severity: AlertSeverity,
        message: str,
        timestamp: datetime,
    ):
        self.alert_id = str(uuid.uuid4())[:8]
        self.patient_id = patient_id
        self.severity = severity
        self.message = message
        self.timestamp = timestamp
        self.acknowledged = False
        self.acknowledged_by: Optional[str] = None
        self.acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "patient_id": self.patient_id,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
        }


class PatientManager:
    """
    Central pipeline orchestrator.

    process_vital() is main entry point:
      Record -> Entropy Engine -> Drug Filter -> Evidence Engine -> Stored State
    """

    def __init__(self, config: Optional[AppConfig] = None):
        if config is None:
            config = load_config()
        self.config = config

        # Core engines
        self.entropy_engine = EntropyEngine(config)
        self.drug_db = DrugDatabase(config.drug_filter.drug_database_path)
        self.drug_filter = DrugFilter(self.drug_db, config)
        self.evidence_engine = EvidenceEngine(config)
        self.evidence_engine.build()

        # Per-patient drug tracking
        self.patient_drugs: Dict[str, List[DrugEffect]] = {}

        # State history per patient (for history endpoint + charts)
        self.state_history: Dict[str, deque] = {}
        self.max_history = 720  # ~12 hours at 1/min

        # Latest state cache (for quick GET)
        self.latest_states: Dict[str, PatientState] = {}

        # Alert management
        self.alerts: List[StoredAlert] = []
        self._previous_severities: Dict[str, AlertSeverity] = {}

        # Phase 3: Alarm fatigue tracking
        self.alarm_tracker = AlarmFatigueTracker()

        # Phase 4: Cross-vital correlation analysis
        self.cross_vital = CrossVitalAnalyzer(
            window_size=config.entropy_engine.window_size
        )

        # Timing
        self.start_time = datetime.utcnow()
        self.total_records_processed = 0

    # ------------------------------------------
    # Main pipeline
    # ------------------------------------------

    def process_vital(self, record: VitalSignRecord) -> PatientState:
        """
        Process a single vital-sign record through the full 3-layer pipeline.

        Returns complete PatientState ready for API/WebSocket delivery.
        """
        # Layer 1: Entropy Engine
        state = self.entropy_engine.process_vital(record)

        # Layer 2: Drug Filter (only if patient has drugs and is past warmup)
        active_drugs = self.patient_drugs.get(record.patient_id, [])
        if active_drugs and not state.calibrating:
            state.active_drugs = active_drugs
            baselines = self.entropy_engine.get_vital_baselines(record.patient_id)
            state = self.drug_filter.apply(state, baselines)

        # Layer 3: Evidence Engine (only on WARNING or CRITICAL)
        if (
            not state.calibrating
            and state.alert.active
            and state.alert.severity
            in (AlertSeverity.WARNING, AlertSeverity.CRITICAL)
        ):
            interventions = self.evidence_engine.query(state)
            state.interventions = interventions

        # Phase 1: Traditional alarm comparison
        traditional_alarm = False
        traditional_alarm_vitals = []
        for vname, (low, high) in _TRADITIONAL_THRESHOLDS.items():
            val = getattr(record, vname, None)
            if val is not None and (val < low or val > high):
                traditional_alarm = True
                traditional_alarm_vitals.append(vname)
        state.traditional_alarm = traditional_alarm
        state.traditional_alarm_vitals = traditional_alarm_vitals

        # Phase 2: Compute clinical scores
        state.clinical_scores = ClinicalScores.compute_all(
            hr=record.heart_rate,
            rr=record.resp_rate,
            spo2=record.spo2,
            bp_sys=record.bp_systolic,
            temp=record.temperature,
        )

        # Phase 4: Cross-vital correlation analysis
        self.cross_vital.update(record.patient_id, {
            "heart_rate": record.heart_rate,
            "spo2": record.spo2,
            "bp_systolic": record.bp_systolic,
            "bp_diastolic": record.bp_diastolic,
            "resp_rate": record.resp_rate,
        })
        if not state.calibrating:
            state.cross_correlations = (
                self.cross_vital.compute_correlations(record.patient_id)
            )
            summary = self.cross_vital.get_decoupling_summary(
                record.patient_id
            )
            if summary.get("clinical_alert"):
                state.decoupling_alerts = [summary["clinical_alert"]]
            else:
                state.decoupling_alerts = []

        # Phase 3: Track alarm fatigue comparison
        if not state.calibrating:
            drug_masked = False
            if hasattr(state.alert, 'drug_masked'):
                drug_masked = state.alert.drug_masked
            self.alarm_tracker.record_comparison(
                patient_id=record.patient_id,
                record=record,
                chronos_severity=state.alert.severity,
                drug_masked=drug_masked,
            )

        # Store
        self._store_state(state)
        self._track_alert(state)
        self.total_records_processed += 1

        return state

    # ------------------------------------------
    # Drug management
    # ------------------------------------------

    def add_drug(self, patient_id: str, drug: DrugEffect):
        """Register a drug administration event for a patient."""
        if patient_id not in self.patient_drugs:
            self.patient_drugs[patient_id] = []
        self.patient_drugs[patient_id].append(drug)

    def get_patient_drugs(self, patient_id: str) -> List[DrugEffect]:
        """Get all drugs for a patient."""
        return self.patient_drugs.get(patient_id, [])

    # ------------------------------------------
    # State queries
    # ------------------------------------------

    def get_patient_state(self, patient_id: str) -> Optional[PatientState]:
        """Get latest full state for a single patient."""
        return self.latest_states.get(patient_id)

    def get_all_summaries(self) -> List[PatientSummary]:
        """Get ward-view summaries for all patients, sorted by severity."""
        return self.entropy_engine.get_all_summaries()

    def get_patient_history(
        self, patient_id: str, hours: int = 6
    ) -> List[PatientState]:
        """Get state history for a patient (up to hours hours back)."""
        history = self.state_history.get(patient_id, deque())
        n = min(hours * 60, len(history))
        return list(history)[-n:]

    # ------------------------------------------
    # Alert management
    # ------------------------------------------

    def get_active_alerts(self) -> List[dict]:
        """Get all unacknowledged alerts."""
        return [a.to_dict() for a in self.alerts if not a.acknowledged]

    def get_all_alerts(self) -> List[dict]:
        """Get all alerts (active + acknowledged)."""
        return [a.to_dict() for a in self.alerts]

    def acknowledge_alert(
        self, alert_id: str, acknowledged_by: str = "clinician"
    ) -> bool:
        """Acknowledge an alert by ID. Returns True if found."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.utcnow()
                return True
        return False

    # ------------------------------------------
    # System health
    # ------------------------------------------

    def get_health(self) -> dict:
        """System health and status information."""
        return {
            "status": "ok",
            "active_patients": len(self.latest_states),
            "total_records_processed": self.total_records_processed,
            "total_alerts": len(self.alerts),
            "active_alerts": sum(1 for a in self.alerts if not a.acknowledged),
            "evidence_engine_ready": self.evidence_engine.is_ready,
            "evidence_cases": self.evidence_engine.case_count,
            "drugs_in_database": len(self.drug_db),
            "uptime_seconds": int(
                (datetime.utcnow() - self.start_time).total_seconds()
            ),
        }

    @property
    def patient_ids(self) -> List[str]:
        """All patient IDs currently tracked."""
        return self.entropy_engine.get_active_patient_ids()

    # ------------------------------------------
    # Private helpers
    # ------------------------------------------

    def _store_state(self, state: PatientState):
        """Store state in history and latest cache."""
        pid = state.patient_id
        if pid not in self.state_history:
            self.state_history[pid] = deque(maxlen=self.max_history)
        self.state_history[pid].append(state)
        self.latest_states[pid] = state

    def _track_alert(self, state: PatientState):
        """Create a StoredAlert when severity escalates to WARNING or CRITICAL."""
        pid = state.patient_id
        prev = self._previous_severities.get(pid, AlertSeverity.NONE)
        curr = state.alert.severity

        severity_rank = {
            AlertSeverity.NONE: 0,
            AlertSeverity.WATCH: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.CRITICAL: 3,
        }

        # Create alert on escalation to WARNING or CRITICAL
        if severity_rank.get(curr, 0) > severity_rank.get(prev, 0):
            if curr in (AlertSeverity.WARNING, AlertSeverity.CRITICAL):
                self.alerts.append(
                    StoredAlert(
                        patient_id=pid,
                        severity=curr,
                        message=state.alert.message,
                        timestamp=state.timestamp,
                    )
                )

        self._previous_severities[pid] = curr
