"""
Entropy Engine — The core intelligence layer of Project Chronos.

Responsibilities:
  1. Maintain sliding windows of vital signs per patient
  2. Compute SampEn for each vital sign
  3. Compute Composite Entropy Score (CES)
  4. Detect trends (rising/stable/falling)
  5. Classify alert severity
  6. Estimate time to predicted event
  7. Compute MSE on demand (for deep-dive views)
"""

import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, Optional, List

from ..config import AppConfig
from ..models import (
    VitalSignRecord,
    PatientState,
    PatientSummary,
    VitalsState,
    VitalDetail,
    AlertState,
    AlertSeverity,
    TrendDirection,
)
from .sampen import sample_entropy
from .mse import multiscale_entropy
from .normalization import normalize_sampen


# The six vital signs we track
VITAL_NAMES = [
    "heart_rate",
    "spo2",
    "bp_systolic",
    "bp_diastolic",
    "resp_rate",
    "temperature",
]


class PatientWindow:
    """
    Manages sliding windows for a single patient's vital signs.

    Stores raw values, computed entropy history, and CES history.
    All buffers are fixed-size deques that automatically discard old data.
    """

    def __init__(self, window_size: int = 300):
        self.window_size = window_size
        self.total_points = 0
        self.last_timestamp: Optional[datetime] = None

        # Raw vital-sign buffers (one deque per vital)
        self.buffers: Dict[str, deque] = {
            name: deque(maxlen=window_size) for name in VITAL_NAMES
        }

        # Timestamp buffer
        self.timestamps: deque = deque(maxlen=window_size)

        # Normalized entropy history per vital (for trend computation)
        self.entropy_history: Dict[str, deque] = {
            name: deque(maxlen=window_size) for name in VITAL_NAMES
        }

        # Composite Entropy Score history (for time-to-event estimation)
        self.ces_history: deque = deque(maxlen=window_size)

    def add_record(self, record: VitalSignRecord):
        """Append a new vital-sign record to all buffers."""
        self.timestamps.append(record.timestamp)
        self.last_timestamp = record.timestamp

        for name in VITAL_NAMES:
            value = getattr(record, name, None)
            self.buffers[name].append(value)

        self.total_points += 1

    def get_array(self, vital_name: str) -> np.ndarray:
        """Get current window for a vital as a NumPy array (NaN for missing)."""
        return np.array(
            [v if v is not None else np.nan for v in self.buffers[vital_name]],
            dtype=np.float64,
        )

    def get_valid_fraction(self, vital_name: str) -> float:
        """Fraction of non-null values in the current window."""
        buf = self.buffers[vital_name]
        if len(buf) == 0:
            return 0.0
        valid = sum(1 for v in buf if v is not None)
        return valid / len(buf)

    @property
    def window_fill(self) -> float:
        """How full the window is (0.0 to 1.0)."""
        if self.window_size == 0:
            return 0.0
        return min(1.0, self.total_points / self.window_size)

    @property
    def is_warmed_up(self) -> bool:
        """True when we have enough data to compute reliable entropy."""
        return self.total_points >= self.window_size

    @property
    def current_size(self) -> int:
        """Number of data points currently in the window."""
        return len(self.timestamps)


class EntropyEngine:
    """
    Core entropy computation engine for all patients.

    Usage:
        engine = EntropyEngine(config)
        state = engine.process_vital(record)  # returns PatientState
    """

    def __init__(self, config: Optional[AppConfig] = None):
        if config is None:
            config = AppConfig()
        self.config = config.entropy_engine
        self.patients: Dict[str, PatientWindow] = {}

        # Pre-compute weight mapping for quick access
        self._weights = {
            "heart_rate": self.config.weights.heart_rate,
            "spo2": self.config.weights.spo2,
            "bp_systolic": self.config.weights.bp_systolic,
            "bp_diastolic": self.config.weights.bp_diastolic,
            "resp_rate": self.config.weights.resp_rate,
            "temperature": self.config.weights.temperature,
        }

    def get_or_create_window(self, patient_id: str) -> PatientWindow:
        """Get existing patient window or create a new one."""
        if patient_id not in self.patients:
            self.patients[patient_id] = PatientWindow(self.config.window_size)
        return self.patients[patient_id]

    def remove_patient(self, patient_id: str):
        """Remove a patient from tracking."""
        self.patients.pop(patient_id, None)

    def get_active_patient_ids(self) -> List[str]:
        """Return list of all patient IDs currently being tracked."""
        return list(self.patients.keys())

    # ──────────────────────────────────────────────
    # Main processing pipeline
    # ──────────────────────────────────────────────

    def process_vital(self, record: VitalSignRecord) -> PatientState:
        """
        Process a single vital-sign record through the entropy pipeline.

        This is the main entry point called for every incoming data point.

        Returns a complete PatientState with entropy scores, trends, and alerts.
        """
        window = self.get_or_create_window(record.patient_id)
        window.add_record(record)

        # Build base patient state
        state = PatientState(
            patient_id=record.patient_id,
            timestamp=record.timestamp,
            calibrating=not window.is_warmed_up,
            window_fill=window.window_fill,
        )

        # During warmup, just report current values — no entropy
        if not window.is_warmed_up:
            state.vitals = self._build_vitals_current_only(record)
            state.alert = AlertState(
                active=False,
                severity=AlertSeverity.NONE,
                message=f"Calibrating... ({window.total_points}/{self.config.window_size} points collected)",
            )
            return state

        # ── Compute entropy for each vital ──
        vitals_state = VitalsState()
        weighted_sum = 0.0
        total_weight = 0.0

        for vital_name in VITAL_NAMES:
            current_value = getattr(record, vital_name, None)
            valid_fraction = window.get_valid_fraction(vital_name)

            detail = VitalDetail(value=current_value)

            if valid_fraction >= self.config.min_valid_fraction and current_value is not None:
                arr = window.get_array(vital_name)

                # Compute SampEn
                sampen_val = sample_entropy(
                    arr,
                    m=self.config.sampen_m,
                    r_fraction=self.config.sampen_r_fraction,
                )

                if not np.isnan(sampen_val):
                    detail.sampen = round(sampen_val, 4)

                    # Normalize for CES
                    norm_val = normalize_sampen(sampen_val, vital_name)
                    detail.sampen_normalized = round(norm_val, 4) if norm_val is not None else None

                    if norm_val is not None:
                        window.entropy_history[vital_name].append(norm_val)
                        weighted_sum += self._weights[vital_name] * norm_val
                        total_weight += self._weights[vital_name]
                else:
                    detail.sampen = None
                    detail.sampen_normalized = None

                # Compute trend from entropy history
                detail.trend = self._compute_trend(window, vital_name)
            else:
                detail.sampen = None
                detail.sampen_normalized = None

            setattr(vitals_state, vital_name, detail)

        state.vitals = vitals_state

        # ── Compute Composite Entropy Score ──
        if total_weight > 0:
            ces = weighted_sum / total_weight
        else:
            ces = 1.0  # default to healthy if no valid entropy data

        ces = round(ces, 4)
        state.composite_entropy = ces
        state.composite_entropy_raw = ces  # Drug filter will modify composite_entropy later
        window.ces_history.append(ces)

        # ── Classify alert ──
        state.alert = self._classify_alert(ces, vitals_state, window)

        return state

    # ──────────────────────────────────────────────
    # MSE (on-demand, not every tick)
    # ──────────────────────────────────────────────

    def compute_mse_for_patient(self, patient_id: str) -> Dict[str, List[Optional[float]]]:
        """
        Compute Multi-Scale Entropy for all vitals of a patient.

        This is expensive — call only when needed (e.g., single-patient deep dive view).
        NOT called on every vital-sign update.
        """
        window = self.patients.get(patient_id)
        if window is None or not window.is_warmed_up:
            return {}

        results = {}
        for vital_name in VITAL_NAMES:
            if window.get_valid_fraction(vital_name) >= self.config.min_valid_fraction:
                arr = window.get_array(vital_name)
                mse_vals = multiscale_entropy(
                    arr,
                    scales=self.config.mse_scales,
                    m=self.config.sampen_m,
                    r_fraction=self.config.sampen_r_fraction,
                )
                results[vital_name] = mse_vals
            else:
                results[vital_name] = [None] * len(self.config.mse_scales)

        return results

    # ──────────────────────────────────────────────
    # Patient summary (for ward view)
    # ──────────────────────────────────────────────

    def get_patient_summary(self, patient_id: str) -> Optional[PatientSummary]:
        """Get a lightweight summary for the multi-patient ward view."""
        window = self.patients.get(patient_id)
        if window is None:
            return None

        # Get latest values
        latest_vitals = {}
        for name in ["heart_rate", "spo2", "bp_systolic", "resp_rate"]:
            buf = window.buffers[name]
            if len(buf) > 0 and buf[-1] is not None:
                latest_vitals[name] = buf[-1]

        ces = window.ces_history[-1] if len(window.ces_history) > 0 else 1.0
        severity = self._ces_to_severity(ces) if window.is_warmed_up else AlertSeverity.NONE

        return PatientSummary(
            patient_id=patient_id,
            composite_entropy=ces,
            alert_severity=severity,
            heart_rate=latest_vitals.get("heart_rate"),
            spo2=latest_vitals.get("spo2"),
            bp_systolic=latest_vitals.get("bp_systolic"),
            resp_rate=latest_vitals.get("resp_rate"),
            calibrating=not window.is_warmed_up,
            last_update=window.last_timestamp,
        )

    def get_all_summaries(self) -> List[PatientSummary]:
        """Get summaries for all tracked patients, sorted by severity."""
        summaries = []
        for pid in self.patients:
            s = self.get_patient_summary(pid)
            if s is not None:
                summaries.append(s)

        # Sort: CRITICAL first, then WARNING, WATCH, NONE
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.WATCH: 2,
            AlertSeverity.NONE: 3,
        }
        summaries.sort(key=lambda s: (severity_order.get(s.alert_severity, 4), s.composite_entropy))
        return summaries

    # ──────────────────────────────────────────────
    # CES history (for charts)
    # ──────────────────────────────────────────────

    def get_ces_history(self, patient_id: str) -> List[float]:
        """Return the CES history for a patient."""
        window = self.patients.get(patient_id)
        if window is None:
            return []
        return list(window.ces_history)

    def get_entropy_history(self, patient_id: str, vital_name: str) -> List[float]:
        """Return the normalized entropy history for a specific vital."""
        window = self.patients.get(patient_id)
        if window is None:
            return []
        return list(window.entropy_history.get(vital_name, []))

    # ──────────────────────────────────────────────
    # Baselines (for Drug Filter)
    # ──────────────────────────────────────────────

    def get_vital_baselines(self, patient_id: str) -> Dict[str, Optional[float]]:
        """
        Return mean value for each vital from the current sliding window.
        Used by the Drug Filter to determine if a vital change is drug-explained.
        """
        window = self.patients.get(patient_id)
        if window is None:
            return {name: None for name in VITAL_NAMES}

        baselines = {}
        for name in VITAL_NAMES:
            arr = window.get_array(name)
            valid = arr[~np.isnan(arr)]
            baselines[name] = float(np.mean(valid)) if len(valid) > 0 else None
        return baselines

    # ──────────────────────────────────────────────
    # Private: Trend computation
    # ──────────────────────────────────────────────

    def _compute_trend(self, window: PatientWindow, vital_name: str) -> TrendDirection:
        """Compute entropy trend direction using linear regression over recent history."""
        history = window.entropy_history[vital_name]

        # Use up to slope_window points, but need at least 10
        n_points = min(self.config.trend.slope_window, len(history))
        if n_points < 10:
            return TrendDirection.STABLE

        recent = np.array(list(history)[-n_points:])

        # Remove NaN values
        valid_mask = ~np.isnan(recent)
        if np.sum(valid_mask) < 10:
            return TrendDirection.STABLE

        y = recent[valid_mask]
        x = np.arange(len(y), dtype=np.float64)

        # Simple linear regression: slope = cov(x,y) / var(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator < 1e-10:
            return TrendDirection.STABLE

        slope = numerator / denominator

        if slope > self.config.trend.rising_threshold:
            return TrendDirection.RISING
        elif slope < self.config.trend.falling_threshold:
            return TrendDirection.FALLING
        else:
            return TrendDirection.STABLE

    # ──────────────────────────────────────────────
    # Private: Alert classification
    # ──────────────────────────────────────────────

    def _ces_to_severity(self, ces: float) -> AlertSeverity:
        """Map CES value to alert severity."""
        t = self.config.thresholds
        if ces >= t.none:
            return AlertSeverity.NONE
        elif ces >= t.watch:
            return AlertSeverity.WATCH
        elif ces >= t.warning:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.CRITICAL

    def _classify_alert(
        self, ces: float, vitals: VitalsState, window: PatientWindow
    ) -> AlertState:
        """Classify alert severity and generate alert message."""
        severity = self._ces_to_severity(ces)

        # Find contributing vitals (those with falling trends or low normalized entropy)
        contributing = []
        for name in VITAL_NAMES:
            detail = getattr(vitals, name)
            if detail.trend == TrendDirection.FALLING:
                contributing.append(name)
            elif detail.sampen_normalized is not None and detail.sampen_normalized < self.config.thresholds.watch:
                if name not in contributing:
                    contributing.append(name)

        # Generate human-readable message
        message = self._generate_message(severity, contributing, ces)

        # Estimate hours to event
        hours_to_event = None
        if severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL):
            hours_to_event = self._estimate_hours_to_event(window, ces)

        return AlertState(
            active=(severity != AlertSeverity.NONE),
            severity=severity,
            message=message,
            hours_to_predicted_event=hours_to_event,
            contributing_vitals=contributing,
            drug_masked=False,  # Phase 2 (Drug Filter) will modify this
        )

    def _generate_message(
        self, severity: AlertSeverity, contributing: List[str], ces: float
    ) -> str:
        """Generate a plain-language alert message for clinicians."""
        vitals_str = ", ".join(v.replace("_", " ") for v in contributing) if contributing else "multiple vitals"

        if severity == AlertSeverity.NONE:
            return "Normal physiological complexity. No concerns detected."
        elif severity == AlertSeverity.WATCH:
            return (
                f"Reduced complexity detected in {vitals_str}. "
                f"CES: {ces:.2f}. Monitoring closely."
            )
        elif severity == AlertSeverity.WARNING:
            return (
                f"⚠ Significant complexity loss in {vitals_str}. "
                f"CES: {ces:.2f}. Deterioration likely within 4-8 hours."
            )
        else:
            return (
                f"🚨 CRITICAL: Near-complete loss of variability in {vitals_str}. "
                f"CES: {ces:.2f}. Immediate assessment required."
            )

    def _estimate_hours_to_event(
        self, window: PatientWindow, current_ces: float
    ) -> Optional[float]:
        """
        Estimate hours until a critical event based on entropy decline rate.

        Uses linear extrapolation of recent CES trend. This is a rough heuristic —
        the Evidence Engine (Phase 2) will provide more sophisticated prediction.
        """
        history = list(window.ces_history)
        if len(history) < 30:
            return None

        # Use last 60 points (~1 hour at 1/min)
        lookback = min(60, len(history))
        recent = np.array(history[-lookback:])

        valid = recent[~np.isnan(recent)]
        if len(valid) < 15:
            return None

        # Linear regression for slope (CES change per minute)
        x = np.arange(len(valid), dtype=np.float64)
        x_mean = np.mean(x)
        y_mean = np.mean(valid)
        numerator = np.sum((x - x_mean) * (valid - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator < 1e-10:
            return None

        slope = numerator / denominator  # CES change per data point (≈per minute)

        if slope >= 0:
            return None  # not declining

        # Extrapolate: how many minutes until CES reaches 0.10 (deep critical)?
        target = 0.10
        if current_ces <= target:
            return 0.0

        minutes_to_critical = (target - current_ces) / slope
        hours = minutes_to_critical / 60.0

        return round(max(0.0, min(24.0, hours)), 1)

    # ──────────────────────────────────────────────
    # Private: Helpers
    # ──────────────────────────────────────────────

    def _build_vitals_current_only(self, record: VitalSignRecord) -> VitalsState:
        """Build a VitalsState with only current values (no entropy). Used during warmup."""
        vitals = VitalsState()
        for name in VITAL_NAMES:
            value = getattr(record, name, None)
            setattr(vitals, name, VitalDetail(value=value))
        return vitals
