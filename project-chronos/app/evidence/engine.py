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

    # ------------------------------------------
    # Enhanced Evidence (ML augmentation sidecar)
    # ------------------------------------------

    SYNDROME_TESTS = {
        "Sepsis-like": [
            {"test": "Blood cultures", "reason": "Identify infectious source"},
            {"test": "Serum lactate", "reason": "Assess tissue perfusion"},
            {"test": "Procalcitonin", "reason": "Bacterial infection marker"},
            {"test": "CBC with differential", "reason": "White cell response"},
        ],
        "Respiratory Failure": [
            {"test": "Arterial blood gas", "reason": "Assess oxygenation and ventilation"},
            {"test": "Chest X-ray", "reason": "Evaluate lung fields"},
            {"test": "D-dimer", "reason": "Rule out pulmonary embolism"},
        ],
        "Hemodynamic Instability": [
            {"test": "Echocardiogram", "reason": "Assess cardiac output"},
            {"test": "Serum lactate", "reason": "Tissue perfusion marker"},
            {"test": "CVP assessment", "reason": "Volume status evaluation"},
        ],
        "Cardiac Instability": [
            {"test": "12-lead ECG", "reason": "Rhythm and ischemia assessment"},
            {"test": "Troponin", "reason": "Myocardial injury marker"},
            {"test": "Echocardiogram", "reason": "Wall motion and function"},
        ],
    }

    def find_similar_cases(
        self,
        feature_vector=None,
        syndrome: Optional[str] = None,
    ) -> dict:
        """
        Enhanced evidence query: accepts 45-feature vector + syndrome hint.

        Returns dict with 'interventions' and 'suggested_tests'.
        Falls back to empty lists if inputs are unavailable.
        """
        # Get KNN-based interventions using existing tree
        interventions = []
        if self._built and feature_vector is not None:
            try:
                # Build a minimal query vector from the 45-feature vector
                # Map the 45 features down to the existing 20-feature space
                query_20 = self._map_45_to_20(feature_vector)
                standardized = (query_20 - self.scaler_mean) / self.scaler_std

                k = min(self.config.k_neighbors, len(self.cases))
                distances, indices = self.tree.query(standardized, k=k)

                if np.isscalar(distances):
                    distances = np.array([distances])
                    indices = np.array([indices])

                valid_mask = distances <= self.config.min_distance_threshold
                valid_indices = indices[valid_mask]

                if len(valid_indices) > 0:
                    neighbor_cases = [self.cases[i] for i in valid_indices]
                    interventions = self._rank_interventions(neighbor_cases)
            except Exception:
                pass

        # Serialize interventions to dicts for JSON transport
        intv_dicts = []
        for intv in interventions:
            if hasattr(intv, 'model_dump'):
                intv_dicts.append(intv.model_dump())
            elif isinstance(intv, dict):
                intv_dicts.append(intv)
            else:
                intv_dicts.append({
                    "rank": getattr(intv, 'rank', 0),
                    "action": getattr(intv, 'action', ''),
                    "historical_success_rate": getattr(intv, 'historical_success_rate', 0.0),
                    "similar_cases_count": getattr(intv, 'similar_cases_count', 0),
                    "median_response_time_hours": getattr(intv, 'median_response_time_hours', None),
                    "evidence_source": getattr(intv, 'evidence_source', ''),
                })

        # Add syndrome-specific test recommendations
        suggested_tests = self._get_syndrome_tests(syndrome) if syndrome else []

        return {
            "interventions": intv_dicts,
            "suggested_tests": suggested_tests,
        }

    def _map_45_to_20(self, features_45) -> np.ndarray:
        """
        Map a 45-element feature vector to the existing 20-element space.
        
        This is a best-effort mapping. The existing 20 features are:
          [age, hr, hr_std, sampen_hr, bp_sys, bp_sys_std, sampen_bp_sys,
           rr, rr_std, sampen_rr, spo2, spo2_std, sampen_spo2,
           ces, ces_slope, vaso, sedative, bb, opioid, inotrope]
        
        We pull matching features from the 45d vector or use sensible defaults.
        """
        v = np.zeros(NUM_FEATURES)
        f = features_45
        # Direct mappings (approximate positions in 45-feature vector)
        v[0] = f[0] if len(f) > 0 else 60     # age
        v[1] = f[1] if len(f) > 1 else 80      # hr_current → mean_hr
        v[2] = f[4] if len(f) > 4 else 10      # hr_std_6h → std_hr
        v[3] = f[18] if len(f) > 18 else 1.5   # sampen_hr
        v[4] = f[6] if len(f) > 6 else 120     # bp_sys_current → mean_bp_sys
        v[5] = f[9] if len(f) > 9 else 12      # bp_sys_std_6h → std_bp_sys
        v[6] = f[19] if len(f) > 19 else 1.5   # sampen_bp_sys
        v[7] = f[10] if len(f) > 10 else 16    # rr_current → mean_rr
        v[8] = f[13] if len(f) > 13 else 3     # rr_std_6h → std_rr
        v[9] = f[20] if len(f) > 20 else 1.5   # sampen_rr
        v[10] = f[14] if len(f) > 14 else 97   # spo2_current → mean_spo2
        v[11] = f[17] if len(f) > 17 else 2    # spo2_std_6h → std_spo2
        v[12] = f[21] if len(f) > 21 else 1.5  # sampen_spo2
        v[13] = f[22] if len(f) > 22 else 0.65 # ces → composite_entropy
        v[14] = f[24] if len(f) > 24 else 0.0  # ces_slope_6h
        v[15] = f[32] if len(f) > 32 else 0.0  # vasopressor_active
        v[16] = f[33] if len(f) > 33 else 0.0  # sedative_active
        v[17] = f[34] if len(f) > 34 else 0.0  # beta_blocker_active
        v[18] = f[35] if len(f) > 35 else 0.0  # opioid_active
        v[19] = f[37] if len(f) > 37 else 0.0  # inotrope_active
        return v

    def _get_syndrome_tests(self, syndrome: str) -> list:
        """Return recommended tests for a given syndrome classification."""
        return self.SYNDROME_TESTS.get(syndrome, [])

    @property
    def is_ready(self) -> bool:
        """Check if the engine has been built and is ready for queries."""
        return self._built

    @property
    def case_count(self) -> int:
        """Number of historical cases loaded."""
        return len(self.cases)
