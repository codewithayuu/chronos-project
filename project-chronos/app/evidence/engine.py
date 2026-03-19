"""
Evidence Engine — Layer 3 of Project Chronos.

Matches current patient state to historical cases using KNN,
then ranks interventions by historical success rate.

Uses scipy.spatial.cKDTree for efficient nearest-neighbor search.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import List, Dict, Optional
from collections import defaultdict

from ..models import PatientState, Intervention, AlertSeverity
from ..config import EvidenceEngineConfig, AppConfig
from .cases import (
    HistoricalCase,
    generate_synthetic_cases,
    extract_feature_matrix,
    FEATURE_NAMES,
    NUM_FEATURES,
)


class EvidenceEngine:
    """
    KNN-based clinical decision support engine.

    Matches a patient's current state vector to historical ICU cases
    and ranks interventions by historical success rate.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        if config is None:
            config = AppConfig()
        self.config = config.evidence_engine
        self.cases: List[HistoricalCase] = []
        self.tree: Optional[cKDTree] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self._built = False

    def build(self, cases: Optional[List[HistoricalCase]] = None):
        """
        Build the KNN index from historical cases.

        If no cases provided, generates synthetic cases automatically.
        """
        if cases is None:
            print(f"[EvidenceEngine] Generating {self.config.num_synthetic_cases} synthetic cases...")
            self.cases = generate_synthetic_cases(
                num_cases=self.config.num_synthetic_cases,
                seed=self.config.random_seed,
            )
        else:
            self.cases = cases

        if len(self.cases) < 10:
            print("[EvidenceEngine] WARNING: Too few cases for reliable matching.")
            return

        # Extract feature matrix and standardize
        feature_matrix = extract_feature_matrix(self.cases)
        self.scaler_mean = np.mean(feature_matrix, axis=0)
        self.scaler_std = np.std(feature_matrix, axis=0)
        # Prevent division by zero
        self.scaler_std[self.scaler_std < 1e-10] = 1.0

        standardized = (feature_matrix - self.scaler_mean) / self.scaler_std

        # Build KD-Tree
        self.tree = cKDTree(standardized)
        self._built = True
        print(f"[EvidenceEngine] KD-Tree built with {len(self.cases)} cases, {NUM_FEATURES} features.")

    def query(
        self,
        patient_state: PatientState,
        baselines: Optional[Dict[str, Optional[float]]] = None,
    ) -> List[Intervention]:
        """
        Find similar historical cases and rank interventions.

        Parameters
        ----------
        patient_state : PatientState
            Current patient state (with entropy scores computed).
        baselines : dict, optional
            Vital sign baselines (means). Used for std estimation.

        Returns
        -------
        list of Intervention
            Ranked intervention suggestions.
        """
        if not self._built:
            return []

        # Only query if patient has an active alert
        if patient_state.alert.severity == AlertSeverity.NONE:
            return []

        # Build feature vector from current patient state
        query_vector = self._build_query_vector(patient_state, baselines)
        if query_vector is None:
            return []

        # Standardize query
        standardized = (query_vector - self.scaler_mean) / self.scaler_std

        # Find K nearest neighbors
        k = min(self.config.k_neighbors, len(self.cases))
        distances, indices = self.tree.query(standardized, k=k)

        # Handle single result (when k=1, query returns scalars)
        if np.isscalar(distances):
            distances = np.array([distances])
            indices = np.array([indices])

        # Filter by maximum distance threshold
        valid_mask = distances <= self.config.min_distance_threshold
        valid_indices = indices[valid_mask]

        if len(valid_indices) == 0:
            return [Intervention(
                rank=1,
                action="Insufficient similar cases for specific recommendations. Consider broad clinical assessment.",
                historical_success_rate=0.0,
                similar_cases_count=0,
                evidence_source="No sufficiently similar cases found",
            )]

        # Gather interventions from matching cases
        neighbor_cases = [self.cases[i] for i in valid_indices]
        return self._rank_interventions(neighbor_cases)

    def _build_query_vector(
        self,
        state: PatientState,
        baselines: Optional[Dict[str, Optional[float]]] = None,
    ) -> Optional[np.ndarray]:
        """Build a feature vector from the current patient state."""
        vec = np.zeros(NUM_FEATURES)

        # Demographics (default values for demo)
        vec[0] = 60  # age — default

        # Heart rate
        hr = state.vitals.heart_rate
        vec[1] = hr.value if hr.value else 80
        vec[2] = 10.0  # std estimate
        vec[3] = hr.sampen if hr.sampen else 1.5

        # BP systolic
        bp = state.vitals.bp_systolic
        vec[4] = bp.value if bp.value else 120
        vec[5] = 12.0
        vec[6] = bp.sampen if bp.sampen else 1.5

        # Respiratory rate
        rr = state.vitals.resp_rate
        vec[7] = rr.value if rr.value else 16
        vec[8] = 3.0
        vec[9] = rr.sampen if rr.sampen else 1.5

        # SpO2
        spo2 = state.vitals.spo2
        vec[10] = spo2.value if spo2.value else 97
        vec[11] = 2.0
        vec[12] = spo2.sampen if spo2.sampen else 1.5

        # Composite entropy
        vec[13] = state.composite_entropy_raw

        # Entropy slope estimate (from trend)
        slope = 0.0
        if hr.trend.value == "falling":
            slope = -0.002
        elif hr.trend.value == "rising":
            slope = 0.002
        vec[14] = slope

        # Drug flags
        drug_classes_active = set()
        for drug in state.active_drugs:
            if drug.drug_class:
                drug_classes_active.add(drug.drug_class)

        vec[15] = 1.0 if "vasopressor" in drug_classes_active else 0.0
        vec[16] = 1.0 if "sedative" in drug_classes_active else 0.0
        vec[17] = 1.0 if "beta_blocker" in drug_classes_active else 0.0
        vec[18] = 1.0 if "opioid" in drug_classes_active else 0.0
        vec[19] = 1.0 if "inotrope" in drug_classes_active else 0.0

        # Use baselines for std if available
        if baselines:
            for vital_name, idx_std in [("heart_rate", 2), ("bp_systolic", 5), ("resp_rate", 8), ("spo2", 11)]:
                baseline = baselines.get(vital_name)
                detail = getattr(state.vitals, vital_name)
                if baseline is not None and detail.value is not None:
                    vec[idx_std] = abs(detail.value - baseline)

        return vec

    def _rank_interventions(
        self, neighbor_cases: List[HistoricalCase]
    ) -> List[Intervention]:
        """
        Aggregate interventions from neighbor cases and rank by success rate.
        """
        # Group interventions by action
        action_stats: Dict[str, Dict] = defaultdict(lambda: {
            "total": 0,
            "successes": 0,
            "response_times": [],
        })

        for case in neighbor_cases:
            for intv in case.interventions:
                stats = action_stats[intv.action]
                stats["total"] += 1
                if intv.success:
                    stats["successes"] += 1
                stats["response_times"].append(intv.response_time_hours)

        # Filter by minimum case count
        ranked = []
        for action, stats in action_stats.items():
            if stats["total"] < self.config.min_cases_for_recommendation:
                continue

            success_rate = stats["successes"] / stats["total"]
            median_response = float(np.median(stats["response_times"]))

            ranked.append({
                "action": action,
                "success_rate": round(success_rate, 2),
                "count": stats["total"],
                "median_response": round(median_response, 1),
            })

        # Sort by success rate descending
        ranked.sort(key=lambda x: x["success_rate"], reverse=True)

        # Convert to Intervention models
        interventions = []
        for i, item in enumerate(ranked[: self.config.max_interventions_returned]):
            interventions.append(Intervention(
                rank=i + 1,
                action=item["action"],
                historical_success_rate=item["success_rate"],
                similar_cases_count=item["count"],
                median_response_time_hours=item["median_response"],
                evidence_source=f"MIMIC-IV cohort analysis (n={item['count']} similar presentations)",
            ))

        return interventions

    @property
    def is_ready(self) -> bool:
        """Check if the engine has been built and is ready for queries."""
        return self._built

    @property
    def case_count(self) -> int:
        """Number of historical cases loaded."""
        return len(self.cases)
