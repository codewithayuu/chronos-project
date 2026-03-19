"""
Synthetic Historical Case Generator.

Generates realistic ICU case clusters for the Evidence Engine's KNN matching.
In production, these would be pre-computed from MIMIC-IV data.
For the hackathon, we generate them deterministically from seed.

Each case represents a completed ICU stay with:
  - Patient features (vitals, entropy, drugs)
  - Interventions applied
  - Outcome (stabilized, mortality, etc.)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ──────────────────────────────────────────────
# Feature names in the order used for KNN
# ──────────────────────────────────────────────

FEATURE_NAMES = [
    "age",
    "mean_hr", "std_hr", "sampen_hr",
    "mean_bp_sys", "std_bp_sys", "sampen_bp_sys",
    "mean_rr", "std_rr", "sampen_rr",
    "mean_spo2", "std_spo2", "sampen_spo2",
    "composite_entropy", "entropy_slope_6h",
    "on_vasopressor", "on_sedative", "on_beta_blocker",
    "on_opioid", "on_inotrope",
]

NUM_FEATURES = len(FEATURE_NAMES)


@dataclass
class InterventionRecord:
    """An intervention applied in a historical case."""
    action: str
    success: bool               # Did the patient stabilize after this?
    response_time_hours: float  # Time to stabilization


@dataclass
class HistoricalCase:
    """A single historical ICU case for KNN matching."""
    case_id: str
    features: np.ndarray        # Shape: (NUM_FEATURES,)
    deterioration_type: str     # "septic_shock", "respiratory_failure", etc.
    interventions: List[InterventionRecord] = field(default_factory=list)
    mortality: bool = False


# ──────────────────────────────────────────────
# Cluster definitions for synthetic generation
# ──────────────────────────────────────────────

CLUSTER_DEFINITIONS = {
    "septic_shock": {
        "count": 150,
        "feature_means": {
            "age": 62, "mean_hr": 110, "std_hr": 15, "sampen_hr": 0.6,
            "mean_bp_sys": 78, "std_bp_sys": 18, "sampen_bp_sys": 0.5,
            "mean_rr": 26, "std_rr": 6, "sampen_rr": 0.55,
            "mean_spo2": 92, "std_spo2": 4, "sampen_spo2": 0.7,
            "composite_entropy": 0.28, "entropy_slope_6h": -0.003,
            "on_vasopressor": 0.6, "on_sedative": 0.4, "on_beta_blocker": 0.05,
            "on_opioid": 0.3, "on_inotrope": 0.1,
        },
        "feature_stds": {
            "age": 15, "mean_hr": 18, "std_hr": 5, "sampen_hr": 0.25,
            "mean_bp_sys": 12, "std_bp_sys": 6, "sampen_bp_sys": 0.2,
            "mean_rr": 5, "std_rr": 2, "sampen_rr": 0.2,
            "mean_spo2": 3, "std_spo2": 2, "sampen_spo2": 0.25,
            "composite_entropy": 0.12, "entropy_slope_6h": 0.001,
            "on_vasopressor": 0.3, "on_sedative": 0.3, "on_beta_blocker": 0.1,
            "on_opioid": 0.25, "on_inotrope": 0.15,
        },
        "interventions": [
            {"action": "Initiate Norepinephrine infusion (0.05-0.1 mcg/kg/min)", "base_success": 0.78, "response_h": 1.2},
            {"action": "Administer 500mL crystalloid fluid bolus", "base_success": 0.65, "response_h": 0.5},
            {"action": "Start broad-spectrum antibiotics (Piperacillin-Tazobactam)", "base_success": 0.70, "response_h": 2.0},
            {"action": "Insert central venous catheter for CVP monitoring", "base_success": 0.55, "response_h": 0.3},
            {"action": "Obtain arterial blood gas and lactate level", "base_success": 0.50, "response_h": 0.2},
        ],
        "mortality_rate": 0.25,
    },
    "respiratory_failure": {
        "count": 120,
        "feature_means": {
            "age": 58, "mean_hr": 100, "std_hr": 12, "sampen_hr": 0.7,
            "mean_bp_sys": 105, "std_bp_sys": 14, "sampen_bp_sys": 0.65,
            "mean_rr": 32, "std_rr": 8, "sampen_rr": 0.4,
            "mean_spo2": 88, "std_spo2": 5, "sampen_spo2": 0.35,
            "composite_entropy": 0.32, "entropy_slope_6h": -0.0025,
            "on_vasopressor": 0.15, "on_sedative": 0.5, "on_beta_blocker": 0.05,
            "on_opioid": 0.35, "on_inotrope": 0.05,
        },
        "feature_stds": {
            "age": 16, "mean_hr": 15, "std_hr": 4, "sampen_hr": 0.2,
            "mean_bp_sys": 15, "std_bp_sys": 5, "sampen_bp_sys": 0.2,
            "mean_rr": 6, "std_rr": 3, "sampen_rr": 0.15,
            "mean_spo2": 4, "std_spo2": 2, "sampen_spo2": 0.15,
            "composite_entropy": 0.10, "entropy_slope_6h": 0.001,
            "on_vasopressor": 0.2, "on_sedative": 0.3, "on_beta_blocker": 0.1,
            "on_opioid": 0.3, "on_inotrope": 0.1,
        },
        "interventions": [
            {"action": "Prepare for endotracheal intubation and mechanical ventilation", "base_success": 0.72, "response_h": 0.3},
            {"action": "Increase FiO2 to 100% and apply high-flow nasal cannula", "base_success": 0.58, "response_h": 0.2},
            {"action": "Administer 500mL crystalloid fluid bolus", "base_success": 0.45, "response_h": 0.5},
            {"action": "Initiate prone positioning if mechanically ventilated", "base_success": 0.52, "response_h": 1.5},
            {"action": "Obtain chest X-ray and arterial blood gas STAT", "base_success": 0.40, "response_h": 0.2},
        ],
        "mortality_rate": 0.20,
    },
    "cardiac_decompensation": {
        "count": 100,
        "feature_means": {
            "age": 68, "mean_hr": 115, "std_hr": 20, "sampen_hr": 0.45,
            "mean_bp_sys": 88, "std_bp_sys": 20, "sampen_bp_sys": 0.4,
            "mean_rr": 24, "std_rr": 5, "sampen_rr": 0.6,
            "mean_spo2": 91, "std_spo2": 4, "sampen_spo2": 0.55,
            "composite_entropy": 0.25, "entropy_slope_6h": -0.0035,
            "on_vasopressor": 0.3, "on_sedative": 0.2, "on_beta_blocker": 0.25,
            "on_opioid": 0.15, "on_inotrope": 0.45,
        },
        "feature_stds": {
            "age": 12, "mean_hr": 18, "std_hr": 6, "sampen_hr": 0.2,
            "mean_bp_sys": 15, "std_bp_sys": 7, "sampen_bp_sys": 0.18,
            "mean_rr": 4, "std_rr": 2, "sampen_rr": 0.2,
            "mean_spo2": 3, "std_spo2": 2, "sampen_spo2": 0.2,
            "composite_entropy": 0.10, "entropy_slope_6h": 0.0012,
            "on_vasopressor": 0.25, "on_sedative": 0.2, "on_beta_blocker": 0.2,
            "on_opioid": 0.15, "on_inotrope": 0.3,
        },
        "interventions": [
            {"action": "Initiate Dobutamine infusion (2.5-5.0 mcg/kg/min)", "base_success": 0.68, "response_h": 1.0},
            {"action": "Administer IV Furosemide 40mg for diuresis", "base_success": 0.60, "response_h": 0.8},
            {"action": "Administer 250mL crystalloid fluid challenge", "base_success": 0.42, "response_h": 0.4},
            {"action": "Obtain echocardiogram STAT for cardiac function assessment", "base_success": 0.50, "response_h": 0.3},
            {"action": "Initiate Milrinone infusion (0.375 mcg/kg/min)", "base_success": 0.55, "response_h": 1.2},
        ],
        "mortality_rate": 0.30,
    },
    "hemorrhagic_shock": {
        "count": 80,
        "feature_means": {
            "age": 52, "mean_hr": 125, "std_hr": 12, "sampen_hr": 0.5,
            "mean_bp_sys": 72, "std_bp_sys": 20, "sampen_bp_sys": 0.35,
            "mean_rr": 28, "std_rr": 5, "sampen_rr": 0.55,
            "mean_spo2": 93, "std_spo2": 3, "sampen_spo2": 0.65,
            "composite_entropy": 0.22, "entropy_slope_6h": -0.004,
            "on_vasopressor": 0.5, "on_sedative": 0.3, "on_beta_blocker": 0.02,
            "on_opioid": 0.4, "on_inotrope": 0.05,
        },
        "feature_stds": {
            "age": 18, "mean_hr": 15, "std_hr": 5, "sampen_hr": 0.2,
            "mean_bp_sys": 12, "std_bp_sys": 8, "sampen_bp_sys": 0.15,
            "mean_rr": 5, "std_rr": 2, "sampen_rr": 0.18,
            "mean_spo2": 3, "std_spo2": 1.5, "sampen_spo2": 0.2,
            "composite_entropy": 0.08, "entropy_slope_6h": 0.001,
            "on_vasopressor": 0.3, "on_sedative": 0.25, "on_beta_blocker": 0.05,
            "on_opioid": 0.3, "on_inotrope": 0.1,
        },
        "interventions": [
            {"action": "Administer 1000mL crystalloid fluid bolus STAT", "base_success": 0.75, "response_h": 0.3},
            {"action": "Initiate packed RBC transfusion (2 units)", "base_success": 0.72, "response_h": 0.5},
            {"action": "Initiate Norepinephrine infusion (0.1-0.2 mcg/kg/min)", "base_success": 0.62, "response_h": 0.8},
            {"action": "Activate massive transfusion protocol", "base_success": 0.58, "response_h": 0.4},
            {"action": "Obtain STAT hemoglobin and coagulation panel", "base_success": 0.45, "response_h": 0.2},
        ],
        "mortality_rate": 0.22,
    },
    "stable": {
        "count": 50,
        "feature_means": {
            "age": 55, "mean_hr": 78, "std_hr": 8, "sampen_hr": 1.8,
            "mean_bp_sys": 122, "std_bp_sys": 10, "sampen_bp_sys": 1.6,
            "mean_rr": 15, "std_rr": 3, "sampen_rr": 1.7,
            "mean_spo2": 97, "std_spo2": 1.5, "sampen_spo2": 1.5,
            "composite_entropy": 0.78, "entropy_slope_6h": 0.0005,
            "on_vasopressor": 0.05, "on_sedative": 0.1, "on_beta_blocker": 0.15,
            "on_opioid": 0.1, "on_inotrope": 0.02,
        },
        "feature_stds": {
            "age": 18, "mean_hr": 10, "std_hr": 3, "sampen_hr": 0.3,
            "mean_bp_sys": 12, "std_bp_sys": 4, "sampen_bp_sys": 0.3,
            "mean_rr": 3, "std_rr": 1.5, "sampen_rr": 0.3,
            "mean_spo2": 1.5, "std_spo2": 0.8, "sampen_spo2": 0.3,
            "composite_entropy": 0.12, "entropy_slope_6h": 0.001,
            "on_vasopressor": 0.1, "on_sedative": 0.15, "on_beta_blocker": 0.2,
            "on_opioid": 0.15, "on_inotrope": 0.05,
        },
        "interventions": [],
        "mortality_rate": 0.02,
    },
}


def generate_synthetic_cases(
    num_cases: int = 500,
    seed: int = 42,
) -> List[HistoricalCase]:
    """
    Generate deterministic synthetic ICU cases for KNN matching.

    Cases are distributed across clinical clusters with realistic
    feature distributions and intervention outcomes.
    """
    rng = np.random.RandomState(seed)
    cases: List[HistoricalCase] = []
    case_counter = 0

    # Calculate proportional counts
    total_defined = sum(c["count"] for c in CLUSTER_DEFINITIONS.values())
    scale = num_cases / total_defined

    for cluster_name, cluster_def in CLUSTER_DEFINITIONS.items():
        n = max(1, int(cluster_def["count"] * scale))
        means = cluster_def["feature_means"]
        stds = cluster_def["feature_stds"]

        for i in range(n):
            case_counter += 1

            # Generate feature vector with noise around cluster center
            features = np.zeros(NUM_FEATURES)
            for j, fname in enumerate(FEATURE_NAMES):
                mean = means[fname]
                std = stds[fname]
                val = rng.normal(mean, std)

                # Clamp binary flags to [0, 1]
                if fname.startswith("on_"):
                    val = float(rng.random() < mean)
                # Clamp entropy/sampen values
                elif "sampen" in fname or "entropy" in fname:
                    val = max(0.0, val)
                # Clamp physiological ranges
                elif fname == "age":
                    val = max(18, min(100, val))
                elif fname == "mean_spo2":
                    val = max(60, min(100, val))
                elif fname == "mean_hr":
                    val = max(30, min(200, val))

                features[j] = val

            # Generate interventions with probabilistic outcomes
            interventions = []
            for intv_def in cluster_def["interventions"]:
                # Not all cases get all interventions
                if rng.random() < 0.7:
                    success = rng.random() < intv_def["base_success"]
                    response_h = intv_def["response_h"] * rng.uniform(0.5, 2.0)
                    interventions.append(InterventionRecord(
                        action=intv_def["action"],
                        success=success,
                        response_time_hours=round(response_h, 2),
                    ))

            mortality = rng.random() < cluster_def["mortality_rate"]

            cases.append(HistoricalCase(
                case_id=f"CASE-{case_counter:04d}",
                features=features,
                deterioration_type=cluster_name,
                interventions=interventions,
                mortality=mortality,
            ))

    return cases


def extract_feature_matrix(cases: List[HistoricalCase]) -> np.ndarray:
    """Extract the N×D feature matrix from a list of cases."""
    return np.array([c.features for c in cases], dtype=np.float64)
