
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from copy import deepcopy

from ..models import (
    PatientState,
    AlertState,
    AlertSeverity,
    DrugEffect,
)
from ..config import DrugFilterConfig, AppConfig
from .database import DrugDatabase


# Vital names that the drug filter can adjust
FILTERABLE_VITALS = ["heart_rate", "spo2", "bp_systolic", "bp_diastolic", "resp_rate"]


class DrugFilter:
    """
    Drug awareness filter that adjusts entropy scores based on active medications.

    Two core behaviors:
    1. SUPPRESS: If a vital change is explained by a drug → reduce its CES weight
    2. DETECT MASKING: If entropy drops but values stay stable → flag as drug_masked
    """

    def __init__(self, drug_db: DrugDatabase, config: Optional[AppConfig] = None):
        self.drug_db = drug_db
        if config is None:
            config = AppConfig()
        self.config = config.drug_filter
        self.entropy_config = config.entropy_engine

        # Base weights for CES recalculation
        self._base_weights = {
            "heart_rate": config.entropy_engine.weights.heart_rate,
            "spo2": config.entropy_engine.weights.spo2,
            "bp_systolic": config.entropy_engine.weights.bp_systolic,
            "bp_diastolic": config.entropy_engine.weights.bp_diastolic,
            "resp_rate": config.entropy_engine.weights.resp_rate,
            "temperature": config.entropy_engine.weights.temperature,
        }

    def apply(
        self,
        patient_state: PatientState,
        baselines: Dict[str, Optional[float]],
    ) -> PatientState:
        """
        Apply drug awareness filter to a patient state.

        Parameters
        ----------
        patient_state : PatientState
            Current state from the entropy engine (with composite_entropy_raw set).
        baselines : dict
            Mapping of vital_name → baseline mean value for comparison.
            Typically computed from the sliding window.

        Returns
        -------
        PatientState
            Modified state with adjusted CES, alert, and drug masking flags.
        """
        state = deepcopy(patient_state)

        if not state.active_drugs or state.calibrating:
            return state

        current_time = state.timestamp

        # Track which vitals are drug-explained and which are masked
        explained_vitals: List[str] = []
        masked_vitals: List[str] = []
        drug_annotations: List[str] = []

        for drug in state.active_drugs:
            if not self._is_within_effect_window(drug, current_time):
                continue

            drug_entry = self.drug_db.lookup(drug.drug_name)
            if drug_entry is None:
                continue

            affected_vitals = self.drug_db.get_affected_vitals(drug.drug_name)

            for vital_name in affected_vitals:
                if vital_name not in FILTERABLE_VITALS:
                    continue

                vital_detail = getattr(state.vitals, vital_name, None)
                if vital_detail is None or vital_detail.value is None:
                    continue

                baseline = baselines.get(vital_name)
                if baseline is None:
                    continue

                expected_change = self.drug_db.get_expected_change(drug.drug_name, vital_name)
                if expected_change is None:
                    continue

                actual_change = vital_detail.value - baseline

                # Check for masking first - this takes priority over explanation
                if self._is_masking(vital_detail, actual_change, expected_change):
                    # Drug is masking true decline
                    if vital_name not in masked_vitals:
                        masked_vitals.append(vital_name)
                        drug_annotations.append(
                            f"⚠ {drug.drug_name} may be masking decline in {vital_name.replace('_', ' ')}"
                        )
                elif self._is_change_explained(actual_change, expected_change):
                    # Drug explains this vital's behavior
                    if vital_name not in explained_vitals:
                        explained_vitals.append(vital_name)
                        drug_annotations.append(
                            f"{vital_name.replace('_', ' ')} change explained by {drug.drug_name}"
                        )

        # Recalculate CES with adjusted weights
        if len(explained_vitals) == 0:
            # No drug effects to adjust, return original CES
            state.composite_entropy = state.composite_entropy_raw
            adjusted_ces = state.composite_entropy_raw
        else:
            adjusted_ces = self._recalculate_ces(state, explained_vitals)
            state.composite_entropy = adjusted_ces

        # Determine drug masking flag
        drug_masked = len(masked_vitals) > 0
        state.alert.drug_masked = drug_masked

        # Reclassify alert based on adjusted CES
        if drug_masked:
            # If masking detected, use the RAW (unadjusted) CES for severity
            # This ensures we don't hide real deterioration
            effective_ces = state.composite_entropy_raw
        else:
            effective_ces = adjusted_ces

        new_severity = self._ces_to_severity(effective_ces)
        state.alert.severity = new_severity
        state.alert.active = (new_severity != AlertSeverity.NONE)

        # Update message with drug context
        state.alert.message = self._build_message(
            state.alert, drug_annotations, adjusted_ces, state.composite_entropy_raw
        )

        return state

    def _is_within_effect_window(self, drug: DrugEffect, current_time: datetime) -> bool:
        """Check if a drug's effect is currently active based on timing."""
        if drug.start_time is None:
            return True  # If no start time, assume active

        drug_entry = self.drug_db.lookup(drug.drug_name)
        if drug_entry is None:
            return False

        onset_time = drug.start_time + timedelta(minutes=drug_entry.onset_minutes)
        end_time = drug.start_time + timedelta(minutes=drug_entry.onset_minutes + drug_entry.duration_minutes)

        is_active = onset_time <= current_time <= end_time
        return is_active

    def _is_change_explained(self, actual_change: float, expected_change: float) -> bool:
        """
        Check if an actual vital sign change is consistent with the expected drug effect.

        Both must be in the same direction, and actual magnitude must be within
        tolerance of expected magnitude.
        """
        if abs(expected_change) < 0.1:
            return False  # No meaningful expected effect

        # Same direction?
        if actual_change * expected_change < 0:
            return False

        # Within tolerance?
        tolerance = abs(expected_change) * (1.0 + self.config.tolerance_fraction)
        return abs(actual_change) <= tolerance

    def _is_masking(self, vital_detail, actual_change: float, expected_change: float) -> bool:
        """
        Detect drug masking: entropy is dropping but vital values appear stable.

        This happens when a drug props up vital sign values while the underlying
        physiology deteriorates (entropy decreases).
        """
        # Entropy must be low or dropping
        entropy_low = (
            vital_detail.sampen_normalized is not None
            and vital_detail.sampen_normalized < 0.40
        )
        from ..models import TrendDirection
        entropy_falling = vital_detail.trend == TrendDirection.FALLING

        if not (entropy_low or entropy_falling):
            return False

        # For masking, we need:
        # 1. Drug has a meaningful expected effect
        # 2. Actual change is much smaller than expected (values appear stable)
        # 3. But entropy is dropping (underlying deterioration)
        
        if abs(expected_change) < 0.1:
            return False

        # Value appears stable relative to expected drug effect
        value_stable = abs(actual_change) < abs(expected_change) * 0.5

        return value_stable and (entropy_low or entropy_falling)

    def _recalculate_ces(
        self, state: PatientState, explained_vitals: List[str]
    ) -> float:
        """
        Recalculate CES with reduced weights for drug-explained vitals.
        """
        vitals_map = {
            "heart_rate": state.vitals.heart_rate,
            "spo2": state.vitals.spo2,
            "bp_systolic": state.vitals.bp_systolic,
            "bp_diastolic": state.vitals.bp_diastolic,
            "resp_rate": state.vitals.resp_rate,
            "temperature": state.vitals.temperature,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for vital_name, detail in vitals_map.items():
            if detail.sampen_normalized is None:
                continue

            weight = self._base_weights[vital_name]

            # Reduce weight for drug-explained vitals
            if vital_name in explained_vitals:
                weight *= self.config.weight_reduction_factor

            weighted_sum += weight * detail.sampen_normalized
            total_weight += weight

        if total_weight <= 0:
            return state.composite_entropy_raw

        return round(weighted_sum / total_weight, 4)

    def _ces_to_severity(self, ces: float) -> AlertSeverity:
        """Map CES value to alert severity (same thresholds as entropy engine)."""
        t = self.entropy_config.thresholds
        if ces >= t.none:
            return AlertSeverity.NONE
        elif ces >= t.watch:
            return AlertSeverity.WATCH
        elif ces >= t.warning:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.CRITICAL

    def _build_message(
        self,
        alert: AlertState,
        drug_annotations: List[str],
        adjusted_ces: float,
        raw_ces: float,
    ) -> str:
        """Build enriched alert message with drug context."""
        base_msg = alert.message

        if not drug_annotations:
            return base_msg

        drug_context = " | Drug context: " + "; ".join(drug_annotations)

        if alert.drug_masked:
            return (
                f"⚠ DRUG MASKING DETECTED — Raw CES: {raw_ces:.2f}, "
                f"Adjusted CES: {adjusted_ces:.2f}. {drug_context}"
            )

        if abs(adjusted_ces - raw_ces) > 0.01:
            return (
                f"{base_msg} (CES adjusted from {raw_ces:.2f} to {adjusted_ces:.2f} "
                f"due to active medications. {drug_context})"
            )

        return base_msg + drug_context
