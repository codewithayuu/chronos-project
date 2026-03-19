"""
Cross-Vital Correlation Analysis — Organ System Decoupling Detection.

In a healthy body, vital signs are correlated:
  - HR and BP are coupled via baroreflex
  - RR and SpO2 are coupled via respiratory drive
  - BP systolic and diastolic are tightly coupled

When these correlations BREAK DOWN, organs are 'decoupling' —
a sign of systemic failure often preceding cardiac arrest.

This is a NOVEL analytical layer beyond individual vital entropy.
No standard ICU monitoring system tracks inter-vital correlations.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque


# Expected correlations in healthy ICU patients
# Based on published hemodynamic coupling literature
EXPECTED_CORRELATIONS = {
    ("heart_rate", "bp_systolic"): {
        "expected": -0.35,
        "name": "Baroreflex Coupling",
        "organ_systems": ["cardiovascular", "autonomic"],
        "clinical_meaning": (
            "Heart rate and blood pressure should be inversely related "
            "via baroreflex. Decoupling suggests autonomic nervous "
            "system failure — body can no longer regulate blood pressure."
        ),
    },
    ("heart_rate", "resp_rate"): {
        "expected": 0.30,
        "name": "Cardiopulmonary Coupling",
        "organ_systems": ["cardiovascular", "respiratory"],
        "clinical_meaning": (
            "Heart rate and respiratory rate normally co-vary through "
            "shared autonomic regulation. Decoupling suggests independent "
            "organ dysfunction — cardiovascular and respiratory systems "
            "are failing independently."
        ),
    },
    ("resp_rate", "spo2"): {
        "expected": -0.45,
        "name": "Respiratory Drive Coupling",
        "organ_systems": ["respiratory"],
        "clinical_meaning": (
            "Respiratory rate and SpO2 should be inversely related — "
            "the body increases breathing rate to compensate for falling "
            "oxygen. Decoupling suggests respiratory compensation failure."
        ),
    },
    ("bp_systolic", "bp_diastolic"): {
        "expected": 0.80,
        "name": "Vascular Tone Coupling",
        "organ_systems": ["cardiovascular"],
        "clinical_meaning": (
            "Systolic and diastolic BP are normally tightly coupled. "
            "Decoupling (widening pulse pressure or erratic relationship) "
            "suggests loss of vascular tone — often seen in septic shock."
        ),
    },
    ("heart_rate", "spo2"): {
        "expected": -0.20,
        "name": "Cardiac-Oxygenation Coupling",
        "organ_systems": ["cardiovascular", "respiratory"],
        "clinical_meaning": (
            "Mild inverse relationship expected — heart rate rises as "
            "oxygenation falls. Strong decoupling suggests multi-organ "
            "stress with loss of compensatory mechanisms."
        ),
    },
}

# Threshold for declaring a pair 'decoupled'
DECOUPLING_THRESHOLD = 0.5
MIN_DATA_POINTS = 20


class CrossVitalAnalyzer:
    """
    Computes rolling Pearson correlations between vital sign pairs
    and detects organ system decoupling.

    This provides a fundamentally different analytical layer from
    entropy — entropy measures individual signal complexity, while
    cross-correlation measures inter-system coordination.
    """

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.patient_buffers: Dict[str, Dict[str, deque]] = {}

    def update(self, patient_id: str, vitals: Dict[str, Optional[float]]):
        """Add a new set of vital sign values for a patient."""
        if patient_id not in self.patient_buffers:
            self.patient_buffers[patient_id] = {
                name: deque(maxlen=self.window_size)
                for name in [
                    "heart_rate", "spo2", "bp_systolic",
                    "bp_diastolic", "resp_rate",
                ]
            }

        for name, buf in self.patient_buffers[patient_id].items():
            val = vitals.get(name)
            buf.append(val if val is not None else float('nan'))

    def compute_correlations(self, patient_id: str) -> Dict[str, Dict]:
        """
        Compute current correlations and decoupling status
        for all vital sign pairs.

        Returns dict keyed by 'vital1__vital2' with:
          - current: float (-1 to 1) or None
          - expected: float
          - deviation: float (absolute difference)
          - decoupled: bool
          - pair_name: str
          - organ_systems: list of affected systems
          - clinical_meaning: str
          - data_available: bool
        """
        buffers = self.patient_buffers.get(patient_id)
        if buffers is None:
            return {}

        results = {}

        for (v1, v2), info in EXPECTED_CORRELATIONS.items():
            key = f"{v1}__{v2}"

            buf1 = buffers.get(v1)
            buf2 = buffers.get(v2)

            if buf1 is None or buf2 is None:
                results[key] = self._insufficient_result(info)
                continue

            arr1 = np.array(list(buf1), dtype=np.float64)
            arr2 = np.array(list(buf2), dtype=np.float64)

            # Find valid (non-NaN) pairs
            valid = ~(np.isnan(arr1) | np.isnan(arr2))
            n_valid = int(np.sum(valid))

            if n_valid < MIN_DATA_POINTS:
                results[key] = self._insufficient_result(info)
                continue

            a1 = arr1[valid]
            a2 = arr2[valid]

            # Compute Pearson correlation
            std1 = np.std(a1)
            std2 = np.std(a2)

            if std1 < 1e-10 or std2 < 1e-10:
                corr = 0.0
            else:
                corr_matrix = np.corrcoef(a1, a2)
                corr = float(corr_matrix[0, 1])
                if np.isnan(corr):
                    corr = 0.0

            expected = info["expected"]
            deviation = abs(corr - expected)

            # Determine if decoupled
            expected_magnitude = abs(expected)
            threshold = max(
                DECOUPLING_THRESHOLD,
                expected_magnitude * 0.6,
            )
            decoupled = deviation > threshold

            results[key] = {
                "current": round(corr, 3),
                "expected": expected,
                "deviation": round(deviation, 3),
                "decoupled": decoupled,
                "pair_name": info["name"],
                "organ_systems": info["organ_systems"],
                "clinical_meaning": info["clinical_meaning"],
                "data_available": True,
                "data_points": n_valid,
            }

        return results

    def get_decoupling_summary(self, patient_id: str) -> Dict:
        """
        Get a summary of organ system decoupling status.

        Returns:
          - decoupled_pairs: list of decoupled pair names
          - decoupled_count: int
          - total_pairs: int
          - affected_systems: list of organ systems affected
          - decoupling_score: float (0-1, proportion decoupled)
          - clinical_alert: str or None
        """
        correlations = self.compute_correlations(patient_id)

        if not correlations:
            return {
                "decoupled_pairs": [],
                "decoupled_count": 0,
                "total_pairs": 0,
                "affected_systems": [],
                "decoupling_score": 0.0,
                "clinical_alert": None,
            }

        available = {
            k: v for k, v in correlations.items()
            if v.get("data_available", False)
        }
        decoupled = {
            k: v for k, v in available.items()
            if v.get("decoupled", False)
        }

        # Collect affected organ systems
        affected_systems = set()
        decoupled_names = []
        for k, v in decoupled.items():
            decoupled_names.append(v["pair_name"])
            for sys in v.get("organ_systems", []):
                affected_systems.add(sys)

        total = len(available)
        n_decoupled = len(decoupled)
        score = n_decoupled / total if total > 0 else 0.0

        # Generate clinical alert
        clinical_alert = None
        if n_decoupled >= 3:
            clinical_alert = (
                f"CRITICAL: {n_decoupled}/{total} organ system couplings "
                f"have broken down. Multi-organ decoupling detected. "
                f"Affected systems: {', '.join(sorted(affected_systems))}. "
                f"This pattern is associated with impending systemic failure."
            )
        elif n_decoupled >= 2:
            clinical_alert = (
                f"WARNING: {n_decoupled}/{total} organ system couplings "
                f"show decoupling ({', '.join(decoupled_names)}). "
                f"Monitor for multi-organ dysfunction."
            )
        elif n_decoupled == 1:
            clinical_alert = (
                f"WATCH: {decoupled_names[0]} decoupling detected. "
                f"Single-system dysfunction may be developing."
            )

        return {
            "decoupled_pairs": decoupled_names,
            "decoupled_count": n_decoupled,
            "total_pairs": total,
            "affected_systems": sorted(affected_systems),
            "decoupling_score": round(score, 3),
            "clinical_alert": clinical_alert,
        }

    def remove_patient(self, patient_id: str):
        """Remove a patient's correlation buffers."""
        self.patient_buffers.pop(patient_id, None)

    @staticmethod
    def _insufficient_result(info: Dict) -> Dict:
        """Return a result for pairs with insufficient data."""
        return {
            "current": None,
            "expected": info["expected"],
            "deviation": None,
            "decoupled": False,
            "pair_name": info["name"],
            "organ_systems": info["organ_systems"],
            "clinical_meaning": info["clinical_meaning"],
            "data_available": False,
            "data_points": 0,
        }
