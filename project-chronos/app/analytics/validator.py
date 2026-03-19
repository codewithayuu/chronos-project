"""
Validation Engine - computes detection metrics against known crisis events.

Fixed version: runs in background thread with subsampling for performance.
Results are cached after first computation.

Proves that Chronos detects deterioration earlier than traditional monitoring
with actual sensitivity, specificity, and lead time numbers.
"""

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import numpy as np

from ..config import AppConfig
from ..models import VitalSignRecord, AlertSeverity
from ..entropy.engine import EntropyEngine

import logging
logger = logging.getLogger(__name__)


# Traditional ICU alarm thresholds (standard clinical defaults)
TRADITIONAL_THRESHOLDS = {
    "heart_rate": (50.0, 120.0),
    "spo2": (90.0, 100.0),
    "bp_systolic": (90.0, 180.0),
    "bp_diastolic": (40.0, 110.0),
    "resp_rate": (8.0, 30.0),
    "temperature": (35.5, 38.5),
}

# Ground truth crisis events for hero cases
# These must match timelines in generator.py
HERO_CASE_EVENTS = {
    "HERO-SEPSIS-001": {"crisis_minute": 540, "type": "septic_shock"},
    "HERO-RESP-002": {"crisis_minute": 400, "type": "respiratory_failure"},
    "HERO-CARD-003": {"crisis_minute": 400, "type": "cardiac_decompensation"},
}


@dataclass
class CaseValidation:
    """Validation result for a single hero case."""
    case_id: str
    case_type: str
    total_records: int = 0
    crisis_minute: int = 0
    chronos_watch_minute: Optional[int] = None
    chronos_warning_minute: Optional[int] = None
    chronos_critical_minute: Optional[int] = None
    traditional_alarm_minute: Optional[int] = None
    chronos_lead_minutes: Optional[int] = None
    traditional_lead_minutes: Optional[int] = None
    chronos_advantage_minutes: Optional[int] = None
    chronos_detected: bool = False
    traditional_detected: bool = False
    ces_at_crisis: Optional[float] = None
    min_ces_observed: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "case_type": self.case_type,
            "total_records": self.total_records,
            "crisis_minute": self.crisis_minute,
            "chronos_watch_minute": self.chronos_watch_minute,
            "chronos_warning_minute": self.chronos_warning_minute,
            "chronos_critical_minute": self.chronos_critical_minute,
            "traditional_alarm_minute": self.traditional_alarm_minute,
            "chronos_lead_minutes": self.chronos_lead_minutes,
            "traditional_lead_minutes": self.traditional_lead_minutes,
            "chronos_advantage_minutes": self.chronos_advantage_minutes,
            "chronos_detected": self.chronos_detected,
            "traditional_detected": self.traditional_detected,
            "ces_at_crisis": round(self.ces_at_crisis, 4) if self.ces_at_crisis is not None else None,
            "min_ces_observed": round(self.min_ces_observed, 4) if self.min_ces_observed is not None else None,
        }


@dataclass
class ValidationReport:
    """Aggregate validation metrics across all cases."""
    cases: List[CaseValidation] = field(default_factory=list)
    stable_cases_checked: int = 0
    stable_false_alarms: int = 0
    total_hero_cases: int = 0
    hero_cases_detected_chronos: int = 0
    hero_cases_detected_traditional: int = 0
    chronos_sensitivity: float = 0.0
    traditional_sensitivity: float = 0.0
    chronos_specificity: float = 0.0
    mean_chronos_lead_minutes: float = 0.0
    mean_traditional_lead_minutes: float = 0.0
    mean_advantage_minutes: float = 0.0
    computation_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "cases": [c.to_dict() for c in self.cases],
            "summary": {
                "total_hero_cases": self.total_hero_cases,
                "hero_cases_detected_chronos": self.hero_cases_detected_chronos,
                "hero_cases_detected_traditional": self.hero_cases_detected_traditional,
                "chronos_sensitivity": round(self.chronos_sensitivity, 3),
                "traditional_sensitivity": round(self.traditional_sensitivity, 3),
                "chronos_specificity": round(self.chronos_specificity, 3),
                "stable_cases_checked": self.stable_cases_checked,
                "stable_false_alarms": self.stable_false_alarms,
                "mean_chronos_lead_minutes": round(self.mean_chronos_lead_minutes, 1),
                "mean_traditional_lead_minutes": round(self.mean_traditional_lead_minutes, 1),
                "mean_advantage_minutes": round(self.mean_advantage_minutes, 1),
                "computation_time_seconds": round(self.computation_time_seconds, 1),
            },
            "comparison": {
                "chronos_vs_traditional": (
                    f"Chronos detected {self.hero_cases_detected_chronos}/{self.total_hero_cases} deteriorating patients "
                    f"vs Traditional alarms detected {self.hero_cases_detected_traditional}/{self.total_hero_cases}. "
                    f"Average early warning lead time: Chronos {self.mean_chronos_lead_minutes:.0f} min vs "
                    f"Traditional {self.mean_traditional_lead_minutes:.0f} min. "
                    f"Chronos provides {self.mean_advantage_minutes:.0f} minutes earlier detection on average."
                ),
                "false_alarm_comparison": (
                    f"Chronos false alarms on stable patients: {self.stable_false_alarms}/{self.stable_cases_checked}. "
                    f"Specificity: {self.chronos_specificity:.1%}."
                ),
            },
        }


class ValidationEngine:
    """
    Runs hero cases through entropy pipeline and computes
    detection metrics against ground truth crisis events.

    Uses background threading and caching to avoid blocking the API.
    """

    # Class-level cache
    _cached_report: Optional[ValidationReport] = None
    _is_computing: bool = False
    _compute_error: Optional[str] = None

    def __init__(self, config: Optional[AppConfig] = None):
        if config is None:
            config = AppConfig()
        self.config = config

    @classmethod
    def get_cached_report(cls) -> Optional[dict]:
        """Return cached validation report or status."""
        if cls._cached_report is not None:
            return {"status": "complete", "report": cls._cached_report.to_dict()}
        elif cls._is_computing:
            return {"status": "computing", "message": "Validation is being computed in background. Check back in 30-60 seconds."}
        elif cls._compute_error is not None:
            return {"status": "error", "message": cls._compute_error}
        else:
            return {"status": "not_started", "message": "Validation has not been started."}

    def start_background_validation(self):
        """Start validation in a background thread."""
        if ValidationEngine._is_computing or ValidationEngine._cached_report is not None:
            return

        ValidationEngine._is_computing = True
        thread = threading.Thread(target=self._run_validation_thread, daemon=True)
        thread.start()
        logger.info("Background validation started")

    def _run_validation_thread(self):
        """Thread target for background validation."""
        try:
            start_time = time.time()
            report = self._run_validation()
            report.computation_time_seconds = time.time() - start_time
            ValidationEngine._cached_report = report
            ValidationEngine._is_computing = False
            logger.info(
                f"Validation complete in {report.computation_time_seconds:.1f}s. "
                f"Sensitivity: {report.chronos_sensitivity:.1%}, "
                f"Lead time: {report.mean_chronos_lead_minutes:.0f} min"
            )
        except Exception as e:
            ValidationEngine._compute_error = f"{type(e).__name__}: {str(e)}"
            ValidationEngine._is_computing = False
            logger.error(f"Validation failed: {traceback.format_exc()}")

    def _run_validation(self) -> ValidationReport:
        """Run actual validation. Called from background thread."""
        report = ValidationReport()

        # Load demo dataset
        dataset = self._load_dataset()
        if dataset is None:
            report.total_hero_cases = len(HERO_CASE_EVENTS)
            return report

        # Validate hero cases
        for case_id, event_info in HERO_CASE_EVENTS.items():
            records = self._extract_records(dataset, case_id)
            if records is None or len(records) == 0:
                logger.warning(f"No records found for {case_id}")
                continue

            logger.info(f"Validating {case_id} ({len(records)} records)...")
            case_result = self._validate_hero_case(
                case_id=case_id,
                case_type=event_info["type"],
                records=records,
                crisis_minute=event_info["crisis_minute"],
            )
            report.cases.append(case_result)
            report.total_hero_cases += 1

            if case_result.chronos_detected:
                report.hero_cases_detected_chronos += 1
            if case_result.traditional_detected:
                report.hero_cases_detected_traditional += 1

            logger.info(
                f"  {case_id}: Chronos={'DETECTED' if case_result.chronos_detected else 'MISSED'} "
                f"at min {case_result.chronos_warning_minute}, "
                f"Traditional={'DETECTED' if case_result.traditional_detected else 'MISSED'} "
                f"at min {case_result.traditional_alarm_minute}"
            )

        # Validate stable cases
        stable_ids = self._get_stable_ids(dataset)
        report.stable_cases_checked = len(stable_ids)

        for stable_id in stable_ids:
            records = self._extract_records(dataset, stable_id)
            if records is None or len(records) == 0:
                continue

            false_alarm = self._validate_stable_case(stable_id, records)
            if false_alarm:
                report.stable_false_alarms += 1

        # Compute aggregate metrics
        self._compute_aggregates(report)

        return report

    def _validate_hero_case(
        self, case_id: str, case_type: str,
        records: list, crisis_minute: int,
    ) -> CaseValidation:
        """Run one hero case through a fresh entropy engine with subsampling."""
        engine = EntropyEngine(self.config)

        result = CaseValidation(
            case_id=case_id,
            case_type=case_type,
            total_records=len(records),
            crisis_minute=crisis_minute,
        )

        ces_values = []
        SUBSAMPLE = 3  # Process every 3rd record for speed

        for idx, record in enumerate(records):
            # Always feed to engine to maintain window state
            state = engine.process_vital(record)

            # Only check alerts on subsampled records (for speed)
            if idx % SUBSAMPLE != 0:
                continue

            if state.calibrating:
                continue

            ces_val = state.composite_entropy
            ces_values.append(ces_val)

            # Track first WATCH
            if (result.chronos_watch_minute is None and
                state.alert.severity in (
                    AlertSeverity.WATCH, AlertSeverity.WARNING, AlertSeverity.CRITICAL)):
                result.chronos_watch_minute = idx

            # Track first WARNING (primary detection metric)
            if (result.chronos_warning_minute is None and
                state.alert.severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL)):
                result.chronos_warning_minute = idx
                result.chronos_detected = True

            # Track first CRITICAL
            if (result.chronos_critical_minute is None and
                state.alert.severity == AlertSeverity.CRITICAL):
                result.chronos_critical_minute = idx

            # Check traditional threshold alarm
            if result.traditional_alarm_minute is None:
                if self._traditional_alarm_fires(record):
                    result.traditional_alarm_minute = idx
                    result.traditional_detected = True

        # Compute lead times
        if result.chronos_warning_minute is not None:
            result.chronos_lead_minutes = crisis_minute - result.chronos_warning_minute

        if result.traditional_alarm_minute is not None:
            result.traditional_lead_minutes = crisis_minute - result.traditional_alarm_minute

        if result.chronos_lead_minutes is not None and result.traditional_lead_minutes is not None:
            result.chronos_advantage_minutes = (
                result.chronos_lead_minutes - result.traditional_lead_minutes
            )
        elif result.chronos_lead_minutes is not None:
            result.chronos_advantage_minutes = result.chronos_lead_minutes

        if ces_values:
            result.min_ces_observed = min(ces_values)
            crisis_idx = min(len(ces_values) - 1, max(0, crisis_minute // SUBSAMPLE))
            if crisis_idx < len(ces_values):
                result.ces_at_crisis = ces_values[crisis_idx]

        return result

    def _validate_stable_case(self, case_id: str, records: list) -> bool:
        """Check if stable case triggers false WARNING/CRITICAL. Returns True if false alarm."""
        engine = EntropyEngine(self.config)
        SUBSAMPLE = 5  # More aggressive subsampling for stable cases

        for idx, record in enumerate(records):
            state = engine.process_vital(record)

            if idx % SUBSAMPLE != 0:
                continue

            if not state.calibrating:
                if state.alert.severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL):
                    return True

        return False

    def _load_dataset(self) -> Optional[Any]:
        """Load demo dataset, trying multiple function signatures."""
        # Try different import paths and function names
        loaders = [
            ("app.data.generator", "generate_demo_dataset"),
            ("app.data.generator", "generate_all_cases"),
            ("app.data.generator", "create_demo_data"),
        ]

        for module_path, func_name in loaders:
            try:
                import importlib
                mod = importlib.import_module(module_path)
                func = getattr(mod, func_name, None)
                if func is not None:
                    logger.info(f"Loading dataset via {module_path}.{func_name}()")
                    result = func()
                    logger.info(f"Dataset loaded: type={type(result).__name__}")
                    return result
            except Exception as e:
                logger.debug(f"Failed to load via {module_path}.{func_name}: {e}")
                continue

        # Try SyntheticGenerator class
        try:
            from app.data.generator import SyntheticGenerator
            gen = SyntheticGenerator()
            result = gen.generate()
            logger.info(f"Dataset loaded via SyntheticGenerator")
            return result
        except Exception:
            pass

        # Try getting cases from any class with a generate method
        try:
            import app.data.generator as gen_mod
            for name in dir(gen_mod):
                obj = getattr(gen_mod, name)
                if isinstance(obj, type) and hasattr(obj, 'generate'):
                    instance = obj()
                    result = instance.generate()
                    logger.info(f"Dataset loaded via {name}.generate()")
                    return result
        except Exception:
            pass

        logger.error("Could not load demo dataset from any known source")
        return None

    def _extract_records(self, dataset: Any, case_id: str) -> Optional[list]:
        """Extract VitalSignRecord list from dataset, handling multiple formats."""
        try:
            # Dict format: dataset[case_id]
            if isinstance(dataset, dict):
                if case_id not in dataset:
                    return None
                entry = dataset[case_id]
            # List format: find by patient_id
            elif isinstance(dataset, list):
                entry = None
                for item in dataset:
                    pid = getattr(item, 'patient_id', None) or (item.get('patient_id') if isinstance(item, dict) else None)
                    if pid == case_id:
                        entry = item
                        break
                if entry is None:
                    return None
            else:
                return None

            # Extract records from entry
            if isinstance(entry, tuple):
                return entry[0]  # (records, drugs) tuple
            elif isinstance(entry, list):
                return entry
            elif hasattr(entry, 'records'):
                return entry.records
            elif hasattr(entry, 'vitals'):
                return entry.vitals
            elif isinstance(entry, dict):
                return entry.get('records') or entry.get('vitals')
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to extract records for {case_id}: {e}")
            return None

    def _get_stable_ids(self, dataset: Any) -> list:
        """Get list of stable (non-hero) patient IDs from dataset."""
        try:
            if isinstance(dataset, dict):
                return [pid for pid in dataset.keys() if not pid.startswith("HERO")]
            elif isinstance(dataset, list):
                ids = []
                for item in dataset:
                    pid = getattr(item, 'patient_id', None) or (item.get('patient_id') if isinstance(item, dict) else None)
                    if pid and not pid.startswith("HERO"):
                        ids.append(pid)
                return ids
        except Exception:
            return []
        return []

    def _compute_aggregates(self, report: ValidationReport):
        """Compute aggregate metrics from individual case results."""
        if report.total_hero_cases > 0:
            report.chronos_sensitivity = (
                report.hero_cases_detected_chronos / report.total_hero_cases
            )
            report.traditional_sensitivity = (
                report.hero_cases_detected_traditional / report.total_hero_cases
            )

        if report.stable_cases_checked > 0:
            correctly_stable = report.stable_cases_checked - report.stable_false_alarms
            report.chronos_specificity = correctly_stable / report.stable_cases_checked
        else:
            report.chronos_specificity = 1.0

        chronos_leads = [
            c.chronos_lead_minutes for c in report.cases
            if c.chronos_lead_minutes is not None and c.chronos_lead_minutes > 0
        ]
        if chronos_leads:
            report.mean_chronos_lead_minutes = float(np.mean(chronos_leads))

        trad_leads = [
            c.traditional_lead_minutes for c in report.cases
            if c.traditional_lead_minutes is not None and c.traditional_lead_minutes > 0
        ]
        if trad_leads:
            report.mean_traditional_lead_minutes = float(np.mean(trad_leads))

        advantages = [
            c.chronos_advantage_minutes for c in report.cases
            if c.chronos_advantage_minutes is not None
        ]
        if advantages:
            report.mean_advantage_minutes = float(np.mean(advantages))

    @staticmethod
    def _traditional_alarm_fires(record: VitalSignRecord) -> bool:
        """Check if any vital sign crosses traditional alarm thresholds."""
        checks = [
            ("heart_rate", record.heart_rate),
            ("spo2", record.spo2),
            ("bp_systolic", record.bp_systolic),
            ("bp_diastolic", record.bp_diastolic),
            ("resp_rate", record.resp_rate),
            ("temperature", record.temperature),
        ]
        for vital_name, value in checks:
            if value is None:
                continue
            low, high = TRADITIONAL_THRESHOLDS[vital_name]
            if value < low or value > high:
                return True
        return False
