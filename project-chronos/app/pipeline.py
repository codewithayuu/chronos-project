from collections import deque
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from copy import deepcopy
import threading

from .config import AppConfig
from .models import (
    VitalSignRecord,
    PatientState,
    PatientSummary,
    DrugEffect,
    AlertState,
    AlertSeverity,
    Intervention,
)
from .entropy.engine import EntropyEngine
from .drugs.database import DrugDatabase
from .drugs.filter import DrugFilter
from .evidence.engine import EvidenceEngine
from .analytics.clinical_scores import ClinicalScores
from .analytics.alarm_fatigue import AlarmFatigueTracker

try:
    from .data.feature_engineer import FeatureEngineer
    from .ml.predictor import DeteriorationPredictor
    from .ml.classifier import SyndromeClassifier
    from .core.fusion import DecisionFusion
    from .core.detectors import DetectorBank
except ImportError:
    from ._stubs import (StubFeatureEngineer as FeatureEngineer,
                         StubDeteriorationPredictor as DeteriorationPredictor,
                         StubSyndromeClassifier as SyndromeClassifier,
                         StubDecisionFusion as DecisionFusion,
                         StubDetectorBank as DetectorBank)
    print("WARNING: Using stub implementations. ML features disabled.")

class ChronosPipeline:
    """
    Central processing pipeline for Project Chronos.

    Usage:
        pipeline = ChronosPipeline(config)
        state = pipeline.process_vital(record)
    """

    def __init__(self, config: Optional[AppConfig] = None):
        if config is None:
            config = AppConfig()
        self.config = config
        self._start_time = datetime.utcnow()

        # Core components
        self.entropy_engine = EntropyEngine(config)

        # Drug system
        self.drug_db = DrugDatabase(config.drug_filter.drug_database_path)
        self.drug_filter = DrugFilter(self.drug_db, config)

        # Evidence engine
        self.evidence_engine = EvidenceEngine(config)
        self.evidence_engine.build()

        # Per-patient state
        self._patient_drugs: Dict[str, List[DrugEffect]] = {}
        self._state_history: Dict[str, deque] = {}
        self._latest_states: Dict[str, PatientState] = {}

        # Alert management
        self._alerts: Dict[str, Dict] = {}
        self._alert_counter = 0
        
        # Phase 3: Alarm fatigue tracking
        self.alarm_tracker = AlarmFatigueTracker()
        
        # Per-patient ML state storage
        self._ml_states: Dict[str, Dict] = {}

        # NEW: ML augmentation sidecar
        self.feature_engineer = FeatureEngineer()
        models_dir = getattr(config, 'ml_models', {}).get('models_dir', '/data/models') if hasattr(config, 'ml_models') else '/data/models'
        self.deterioration_predictor = DeteriorationPredictor(models_dir=models_dir)
        self.syndrome_classifier = SyndromeClassifier(models_dir=models_dir)
        self.decision_fusion = DecisionFusion()
        self.detectors = DetectorBank()
        
        print(f"ML Status: Predictor={'READY' if getattr(self.deterioration_predictor, 'available', False) else 'UNAVAILABLE'}")
        print(f"ML Status: Classifier={'READY' if getattr(self.syndrome_classifier, 'available', False) else 'UNAVAILABLE'}")

        print("[Pipeline] Initialized — Entropy Engine + Drug Filter + Evidence Engine + ML ready")

    # ──────────────────────────────────────────────
    # Main processing
    # ──────────────────────────────────────────────

    def process_vital(self, record: VitalSignRecord) -> PatientState:
        """
        Process a single vital-sign record through the full pipeline.

        Steps:
          1. Entropy Engine computes SampEn, CES, trends, alert
          2. Drug Filter adjusts CES for active medications
          3. Evidence Engine provides intervention suggestions
          4. State is stored in history

        Returns a complete PatientState.
        """
        pid = record.patient_id

        # Step 1: Entropy Engine
        state = self.entropy_engine.process_vital(record)

        if not state.calibrating:
            # Step 2: Drug Filter
            active_drugs = self.get_active_drugs(pid, record.timestamp)
            if active_drugs:
                state.active_drugs = active_drugs
                baselines = self.entropy_engine.get_vital_baselines(pid)
                state = self.drug_filter.apply(state, baselines)
            
            # Step 3: Evidence Engine (only when alert is active)
            if state.alert.active:
                baselines = self.entropy_engine.get_vital_baselines(pid)
                interventions = self.evidence_engine.query(state, baselines)
                state.interventions = interventions

                # Track alert
                self._track_alert(state)

        # Phase 1: Traditional alarm comparison
        traditional_alarm = False
        traditional_alarm_vitals = []
        _thresholds = {
            "heart_rate": (50.0, 120.0),
            "spo2": (90.0, 100.0),
            "bp_systolic": (90.0, 180.0),
            "bp_diastolic": (40.0, 110.0),
            "resp_rate": (8.0, 30.0),
            "temperature": (35.5, 38.5),
        }
        for _vname, (_low, _high) in _thresholds.items():
            _val = getattr(record, _vname, None)
            if _val is not None and (_val < _low or _val > _high):
                traditional_alarm = True
                traditional_alarm_vitals.append(_vname)
        
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

        # Phase 3: Track alarm fatigue comparison
        if not state.calibrating:
            self.alarm_tracker.record_comparison(
                patient_id=record.patient_id,
                record=record,
                chronos_severity=state.alert.severity,
                drug_masked=state.alert.drug_masked if hasattr(state.alert, 'drug_masked') else False,
            )

        # ============================================================
        # STEPS BELOW ARE NEW — ML AUGMENTATION SIDECAR
        # ============================================================
        
        window = self.entropy_engine.patients.get(pid)
        
        # extract basic entropy_state dict from state
        entropy_state = {
            'ces_adjusted': state.composite_entropy,
            'ces_slope_6h': 0.0, 
            'calibrating': state.calibrating,
            'warmup_complete': not state.calibrating,
            'sampen_hr': state.vitals.heart_rate.sampen_normalized,
            'sampen_bp_sys': state.vitals.bp_systolic.sampen_normalized,
            'sampen_rr': state.vitals.resp_rate.sampen_normalized,
            'sampen_spo2': state.vitals.spo2.sampen_normalized,
            'window_size': window.current_size if window else 0
        }
        
        # extract basic drug_state dict
        active_drugs_dicts = []
        for d in state.active_drugs:
            active_drugs_dicts.append({
                "drug_name": d.drug_name,
                "drug_class": d.drug_class,
                "dose": d.dose,
                "unit": d.unit
            })
        drug_state = {
            'drug_masking': getattr(state.alert, 'drug_masked', False),
            'active_drugs': active_drugs_dicts
        }

        # Step 4: Feature Engineering
        try:
            window_dict = {
                'hr': list(window.buffers['heart_rate']) if window else [],
                'spo2': list(window.buffers['spo2']) if window else [],
                'bp_sys': list(window.buffers['bp_systolic']) if window else [],
                'bp_dia': list(window.buffers['bp_diastolic']) if window else [],
                'rr': list(window.buffers['resp_rate']) if window else [],
                'temp': list(window.buffers['temperature']) if window else []
            }
            
            # extract basic demographics
            demographics = {'age': 60, 'sex': 'unknown', 'weight_kg': 70.0}

            features = self.feature_engineer.compute_features(
                vitals_window=window_dict,
                entropy_state=entropy_state,
                drug_state=drug_state,
                demographics=demographics
            )
        except Exception as e:
            print(f"Feature engineering failed: {e}")
            features = None

        # Step 5: NEW — ML Predictions
        ml_predictions = {'deterioration': None, 'syndrome': None, 'warmup_mode': True}

        if features is not None:
            warmup_complete = entropy_state.get('warmup_complete', False)
            
            if warmup_complete:
                det_result = self.deterioration_predictor.predict(features)
                syn_result = self.syndrome_classifier.predict(features)
                ml_predictions = {
                    'deterioration': det_result,
                    'syndrome': syn_result,
                    'warmup_mode': False
                }
            else:
                features_imputed = self.feature_engineer.impute_warmup(features)
                det_result = self.deterioration_predictor.predict(features_imputed)
                ml_predictions = {
                    'deterioration': det_result,
                    'syndrome': None,
                    'warmup_mode': True
                }

        # Step 6: NEW — Enhanced Evidence
        syndrome_hint = None
        if ml_predictions.get('syndrome') and not ml_predictions['syndrome'].get('inconclusive', True):
            syndrome_hint = ml_predictions['syndrome'].get('primary_syndrome')

        enhanced_evidence = self.evidence_engine.find_similar_cases(
            feature_vector=features,
            syndrome=syndrome_hint
        )

        # Step 7: NEW — Decision Fusion
        det = ml_predictions.get('deterioration') or {}
        fusion_result = self.decision_fusion.fuse(
            ces_adjusted=state.composite_entropy,
            ces_slope_6h=entropy_state.get('ces_slope_6h', 0.0),
            ml_risk_1h=det.get('risk_1h'),
            ml_risk_4h=det.get('risk_4h'),
            ml_risk_8h=det.get('risk_8h'),
            drug_masking=drug_state.get('drug_masking', False),
            news2_score=state.clinical_scores.get('news2', {}).get('score', 0) if state.clinical_scores else 0
        )

        # Step 8: NEW — Detectors
        vitals_dict = record.model_dump() if hasattr(record, 'model_dump') else {}
        detector_results = self.detectors.run_all(
            entropy_state=entropy_state,
            drug_state=drug_state,
            vitals=vitals_dict,
            ml_predictions=ml_predictions,
            fusion=fusion_result
        )

        # Store in ml_state
        self._ml_states[pid] = {
            'ml_predictions': ml_predictions,
            'fusion': fusion_result,
            'detectors': detector_results,
            'recommendations': {
                'interventions': enhanced_evidence.get('interventions', []),
                'suggested_tests': enhanced_evidence.get('suggested_tests', [])
            }
        }

        # Step 10: Store state
        self._latest_states[pid] = state
        self._store_history(pid, state)

        return state

    # ──────────────────────────────────────────────
    # Drug management
    # ──────────────────────────────────────────────

    def add_drug(self, patient_id: str, drug: DrugEffect) -> None:
        """Add a drug administration event to a patient's list."""
        if patient_id not in self._patient_drugs:
            self._patient_drugs[patient_id] = []
        self._patient_drugs[patient_id].append(drug)

    def get_active_drugs(
        self, patient_id: str, current_time: Optional[datetime] = None
    ) -> List[DrugEffect]:
        """Get list of drugs for a patient. Includes all administered drugs."""
        return self._patient_drugs.get(patient_id, [])

    def get_patient_drugs(self, patient_id: str) -> List[DrugEffect]:
        """Get all drugs for a patient (for API)."""
        return self._patient_drugs.get(patient_id, [])

    # ──────────────────────────────────────────────
    # State access (for API)
    # ──────────────────────────────────────────────

    def get_patient_state(self, patient_id: str) -> Optional[PatientState]:
        """Get the latest state for a patient."""
        return self._latest_states.get(patient_id)

    def get_ml_state(self, patient_id: str) -> Dict[str, Any]:
        """Get the latest ML state for a patient."""
        return self._ml_states.get(patient_id, {})

    def get_all_patient_ids(self) -> List[str]:
        """Get all tracked patient IDs."""
        return list(self._latest_states.keys())

    def get_all_summaries(self) -> List[PatientSummary]:
        """Get sorted summaries for all patients (ward view)."""
        return self.entropy_engine.get_all_summaries()

    def get_patient_history(
        self, patient_id: str, hours: float = 6.0
    ) -> List[PatientState]:
        """Get historical states for a patient within a time window."""
        history = self._state_history.get(patient_id)
        if not history:
            return []

        if hours <= 0:
            return list(history)

        cutoff_minutes = int(hours * 60)
        # Return last N entries (each entry ≈ 1 minute)
        return list(history)[-cutoff_minutes:]

    def remove_patient(self, patient_id: str) -> None:
        """Remove a patient from all tracking."""
        self.entropy_engine.remove_patient(patient_id)
        self._patient_drugs.pop(patient_id, None)
        self._state_history.pop(patient_id, None)
        self._latest_states.pop(patient_id, None)
        # Remove alerts for this patient
        to_remove = [
            aid for aid, alert in self._alerts.items()
            if alert.get("patient_id") == patient_id
        ]
        for aid in to_remove:
            del self._alerts[aid]

    # ──────────────────────────────────────────────
    # Alert management
    # ──────────────────────────────────────────────

    def get_all_alerts(self) -> List[Dict]:
        """Get all currently active alerts across all patients."""
        active = []
        for alert_id, alert_data in self._alerts.items():
            pid = alert_data["patient_id"]
            state = self._latest_states.get(pid)
            if state and state.alert.active:
                active.append({
                    "alert_id": alert_id,
                    "patient_id": pid,
                    "severity": state.alert.severity.value,
                    "message": state.alert.message,
                    "hours_to_event": state.alert.hours_to_predicted_event,
                    "drug_masked": state.alert.drug_masked,
                    "contributing_vitals": state.alert.contributing_vitals,
                    "acknowledged": alert_data.get("acknowledged", False),
                    "acknowledged_by": alert_data.get("acknowledged_by"),
                    "timestamp": state.timestamp.isoformat(),
                })
        return active

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert. Returns True if alert was found."""
        if alert_id in self._alerts:
            self._alerts[alert_id]["acknowledged"] = True
            self._alerts[alert_id]["acknowledged_by"] = acknowledged_by
            return True
        return False

    def _track_alert(self, state: PatientState) -> None:
        """Create or update alert tracking for a patient."""
        pid = state.patient_id
        # Check if we already have an active alert for this patient
        existing_id = None
        for aid, adata in self._alerts.items():
            if adata["patient_id"] == pid and not adata.get("acknowledged", False):
                existing_id = aid
                break

        if existing_id is None:
            self._alert_counter += 1
            alert_id = f"ALERT-{self._alert_counter:04d}"
            self._alerts[alert_id] = {
                "patient_id": pid,
                "acknowledged": False,
                "acknowledged_by": None,
                "created_at": state.timestamp.isoformat(),
            }

    # ──────────────────────────────────────────────
    # History storage
    # ──────────────────────────────────────────────

    def _store_history(self, patient_id: str, state: PatientState) -> None:
        """Store a state snapshot in the patient's history buffer."""
        if patient_id not in self._state_history:
            self._state_history[patient_id] = deque(maxlen=1440)  # 24 hours at 1/min
        self._state_history[patient_id].append(deepcopy(state))

    # ──────────────────────────────────────────────
    # System info
    # ──────────────────────────────────────────────

    def get_system_health(self) -> Dict:
        """Get system health status."""
        return {
            "status": "ok",
            "active_patients": len(self._latest_states),
            "uptime_seconds": int((datetime.utcnow() - self._start_time).total_seconds()),
            "evidence_engine_ready": self.evidence_engine.is_ready,
            "evidence_engine_cases": self.evidence_engine.case_count,
            "drug_database_size": len(self.drug_db),
            "total_alerts": len(self._alerts),
            "active_alerts": sum(
                1 for a in self._alerts.values() if not a.get("acknowledged", False)
            ),
        }
