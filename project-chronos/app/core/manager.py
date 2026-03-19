
import uuid
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque

import numpy as np

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

# ── ML Augmentation Sidecar Imports ──────────────────────
# Try to import real implementations; fall back to stubs.
try:
    from ..data.feature_engineer import FeatureEngineer
except ImportError:
    from .._stubs import StubFeatureEngineer as FeatureEngineer

try:
    from ..ml.predictor import DeteriorationPredictor
except ImportError:
    from .._stubs import StubDeteriorationPredictor as DeteriorationPredictor

try:
    from ..ml.classifier import SyndromeClassifier
except ImportError:
    from .._stubs import StubSyndromeClassifier as SyndromeClassifier

try:
    from ..core.fusion import DecisionFusion
except ImportError:
    from .._stubs import StubDecisionFusion as DecisionFusion

try:
    from ..core.detectors import DetectorBank
except ImportError:
    from .._stubs import StubDetectorBank as DetectorBank


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

        # ── ML Augmentation Sidecar ──────────────────────
        # Read ml_models config if present (Person 2 adds it to config.yml)
        raw_cfg = {}
        try:
            from pathlib import Path
            import yaml
            cfg_path = Path("config.yml")
            if cfg_path.exists():
                with open(cfg_path) as f:
                    raw_cfg = yaml.safe_load(f) or {}
        except Exception:
            pass

        ml_cfg = raw_cfg.get("ml_models", {})
        models_dir = ml_cfg.get("models_dir", "data/models")

        self.feature_engineer = FeatureEngineer()
        self.deterioration_predictor = DeteriorationPredictor(models_dir=models_dir)
        self.syndrome_classifier = SyndromeClassifier(models_dir=models_dir)
        self.decision_fusion = DecisionFusion()
        self.detector_bank = DetectorBank()

        # Per-patient vitals window for feature engineering
        self._vitals_windows: Dict[str, deque] = {}

        # Log ML status
        pred_ok = getattr(self.deterioration_predictor, 'available', False)
        clf_ok = getattr(self.syndrome_classifier, 'available', False)
        print(f"[ML] Predictor={'READY' if pred_ok else 'UNAVAILABLE'}  "
              f"Classifier={'READY' if clf_ok else 'UNAVAILABLE'}")

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

        # ============================================================
        # ML AUGMENTATION SIDECAR — Steps 4-9
        # Every step is wrapped in try/except so the existing
        # pipeline is NEVER broken by ML failures.
        # ============================================================

        # Track vitals for feature engineering window
        self._track_vitals_window(record)

        # Prepare default ML state fields
        ml_predictions: Dict[str, Any] = {
            "deterioration_risk": None,
            "syndrome": None,
            "warmup_mode": True,
        }
        fusion_result: Dict[str, Any] = {
            "final_risk_score": 0,
            "final_severity": state.alert.severity.value if state.alert else "NONE",
            "time_to_event_estimate": "Unknown",
            "component_risks": {},
            "ml_available": False,
            "override_applied": None,
            "disagreement": None,
        }
        detector_results: List[Dict] = []
        recommendations: Dict[str, Any] = {
            "interventions": [],
            "suggested_tests": [],
        }
        features = None

        if not state.calibrating:
            # Step 4: Feature Engineering
            try:
                entropy_state_dict = self._build_entropy_state(state)
                drug_state_dict = self._build_drug_state(state)
                demographics = self._get_demographics(record.patient_id)
                vitals_window = self._get_vitals_window(record.patient_id)

                features = self.feature_engineer.compute_features(
                    vitals_window=vitals_window,
                    entropy_state=entropy_state_dict,
                    drug_state=drug_state_dict,
                    demographics=demographics,
                )
            except Exception as e:
                # Feature engineering failed — features stays None
                pass

            # Step 5: ML Predictions
            try:
                if features is not None:
                    warmup_complete = not state.calibrating
                    if warmup_complete:
                        det_result = self.deterioration_predictor.predict(features)
                        syn_result = self.syndrome_classifier.predict(features)
                        ml_predictions = {
                            "deterioration_risk": det_result,
                            "syndrome": syn_result,
                            "warmup_mode": False,
                        }
                    else:
                        features_imputed = self.feature_engineer.impute_warmup(features)
                        det_result = self.deterioration_predictor.predict(features_imputed)
                        ml_predictions = {
                            "deterioration_risk": det_result,
                            "syndrome": None,
                            "warmup_mode": True,
                        }
            except Exception:
                pass

            # Step 6: Enhanced Evidence (pass syndrome hint)
            try:
                syndrome_hint = None
                syn_data = ml_predictions.get("syndrome")
                if syn_data and not syn_data.get("inconclusive", True):
                    syndrome_hint = syn_data.get("primary_syndrome")

                enhanced_evidence = self.evidence_engine.find_similar_cases(
                    feature_vector=features,
                    syndrome=syndrome_hint,
                )
                recommendations = {
                    "interventions": enhanced_evidence.get("interventions", []),
                    "suggested_tests": enhanced_evidence.get("suggested_tests", []),
                }
            except Exception:
                # Method might not exist yet on EvidenceEngine
                pass

            # Step 7: Decision Fusion
            try:
                det_data = ml_predictions.get("deterioration_risk") or {}
                ces_val = state.composite_entropy if hasattr(state, 'composite_entropy') else 0.65

                # Estimate CES slope from history
                ces_slope = self._estimate_ces_slope(record.patient_id)

                drug_masked = False
                if hasattr(state.alert, 'drug_masked'):
                    drug_masked = state.alert.drug_masked

                news2 = 0
                if state.clinical_scores:
                    news2 = state.clinical_scores.get("news2", {}).get("score", 0)

                fusion_result = self.decision_fusion.fuse(
                    ces_adjusted=ces_val,
                    ces_slope_6h=ces_slope,
                    ml_risk_1h=det_data.get("risk_1h"),
                    ml_risk_4h=det_data.get("risk_4h"),
                    ml_risk_8h=det_data.get("risk_8h"),
                    drug_masking=drug_masked,
                    news2_score=news2,
                )
            except Exception:
                pass

            # Step 8: Detectors
            try:
                entropy_state_dict = self._build_entropy_state(state)
                drug_state_dict = self._build_drug_state(state)
                vitals_dict = {
                    "hr": record.heart_rate,
                    "bp_sys": record.bp_systolic,
                    "bp_dia": record.bp_diastolic,
                    "rr": record.resp_rate,
                    "spo2": record.spo2,
                    "temp": record.temperature,
                }
                detector_results = self.detector_bank.run_all(
                    entropy_state=entropy_state_dict,
                    drug_state=drug_state_dict,
                    vitals=vitals_dict,
                    ml_predictions=ml_predictions,
                    fusion=fusion_result,
                )
            except Exception:
                pass

        # Step 9: Attach ML fields to patient state via extra dict
        # We store these as extra attributes since PatientState might
        # not have them yet (Person 2 adds to models.py).
        state._ml_predictions = ml_predictions
        state._fusion = fusion_result
        state._detectors = detector_results
        state._recommendations = recommendations

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
    # ML Augmentation Helpers
    # ------------------------------------------

    def _track_vitals_window(self, record: VitalSignRecord):
        """Keep a rolling window of raw vital values per patient for feature engineering."""
        pid = record.patient_id
        if pid not in self._vitals_windows:
            self._vitals_windows[pid] = deque(maxlen=360)  # 6 hours at 1/min
        self._vitals_windows[pid].append({
            "hr": record.heart_rate,
            "bp_sys": record.bp_systolic,
            "bp_dia": record.bp_diastolic,
            "rr": record.resp_rate,
            "spo2": record.spo2,
            "temp": record.temperature,
            "timestamp": record.timestamp,
        })

    def _get_vitals_window(self, patient_id: str) -> dict:
        """Build rolling vital-sign arrays from stored window."""
        window = self._vitals_windows.get(patient_id, deque())
        result: Dict[str, list] = {
            "hr": [], "bp_sys": [], "bp_dia": [],
            "rr": [], "spo2": [], "temp": [],
        }
        for entry in window:
            for key in result:
                val = entry.get(key)
                result[key].append(val if val is not None else 0.0)
        return result

    def _get_demographics(self, patient_id: str) -> dict:
        """Return demographics for a patient. Defaults for demo."""
        # In a production system this would come from EHR/ADT
        return {"age": 60, "sex": "M", "weight_kg": 75.0}

    def _build_entropy_state(self, state: PatientState) -> dict:
        """Extract entropy-related fields into a dict for ML components."""
        hr_detail = state.vitals.heart_rate
        bp_detail = state.vitals.bp_systolic
        rr_detail = state.vitals.resp_rate
        spo2_detail = state.vitals.spo2

        return {
            "ces_adjusted": state.composite_entropy,
            "ces_raw": state.composite_entropy_raw,
            "ces_slope_6h": self._estimate_ces_slope(state.patient_id),
            "warmup_complete": not state.calibrating,
            "sampen_hr": hr_detail.sampen if hr_detail else None,
            "sampen_bp_sys": bp_detail.sampen if bp_detail else None,
            "sampen_rr": rr_detail.sampen if rr_detail else None,
            "sampen_spo2": spo2_detail.sampen if spo2_detail else None,
            "window_size": int(state.window_fill * 300) if state.window_fill else 0,
        }

    def _build_drug_state(self, state: PatientState) -> dict:
        """Extract drug context into a dict for ML components."""
        drug_masked = False
        if hasattr(state.alert, 'drug_masked'):
            drug_masked = state.alert.drug_masked

        active_drugs = []
        for d in (state.active_drugs or []):
            active_drugs.append({
                "drug_name": d.drug_name,
                "drug_class": d.drug_class,
                "dose": d.dose,
                "unit": d.unit,
            })

        return {
            "drug_masking": drug_masked,
            "active_drugs": active_drugs,
        }

    def _estimate_ces_slope(self, patient_id: str) -> float:
        """Estimate the 6-hour CES slope from history."""
        history = self.state_history.get(patient_id, deque())
        if len(history) < 10:
            return 0.0

        # Use last 360 points (6h) or whatever is available
        recent = list(history)[-360:]
        ces_values = [s.composite_entropy for s in recent if hasattr(s, 'composite_entropy')]
        if len(ces_values) < 10:
            return 0.0

        # Simple linear regression slope
        n = len(ces_values)
        x = np.arange(n, dtype=float)
        y = np.array(ces_values, dtype=float)
        x_mean = x.mean()
        y_mean = y.mean()
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        if abs(denominator) < 1e-12:
            return 0.0
        return float(numerator / denominator)

    def get_ml_state(self, patient_id: str) -> dict:
        """Get the ML augmentation state for a patient (for API endpoints)."""
        state = self.latest_states.get(patient_id)
        if state is None:
            return {
                "ml_predictions": {"warmup_mode": True},
                "fusion": {"final_risk_score": 0, "final_severity": "NONE"},
                "detectors": [],
                "recommendations": {"interventions": [], "suggested_tests": []},
            }
        return {
            "ml_predictions": getattr(state, '_ml_predictions', {"warmup_mode": True}),
            "fusion": getattr(state, '_fusion', {"final_risk_score": 0, "final_severity": "NONE"}),
            "detectors": getattr(state, '_detectors', []),
            "recommendations": getattr(state, '_recommendations', {"interventions": [], "suggested_tests": []}),
        }

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
