"""
Temporary stubs for Person 1 and Person 2 classes.
These provide safe fallbacks when real ML implementations are not yet available.
Delete this file once real implementations from Person 1 & Person 2 are integrated.
"""

import numpy as np


class StubFeatureEngineer:
    """Stub for app.data.feature_engineer.FeatureEngineer"""

    def __init__(self, population_stats_path: str = "data/ml/population_stats.json"):
        pass

    def compute_features(self, vitals_window, entropy_state, drug_state, demographics):
        """Returns a zero 45-feature vector."""
        return np.zeros(45)

    def impute_warmup(self, features):
        """Returns features unchanged."""
        return features


class StubDeteriorationPredictor:
    """Stub for app.ml.predictor.DeteriorationPredictor"""

    available = False

    def __init__(self, models_dir: str = "data/models"):
        pass

    def predict(self, features):
        """Returns None — ML unavailable."""
        return None


class StubSyndromeClassifier:
    """Stub for app.ml.classifier.SyndromeClassifier"""

    available = False

    def __init__(self, models_dir: str = "data/models"):
        pass

    def predict(self, features):
        """Returns None — ML unavailable."""
        return None


class StubDecisionFusion:
    """Stub for app.core.fusion.DecisionFusion — entropy-only fallback."""

    def fuse(
        self,
        ces_adjusted,
        ces_slope_6h,
        ml_risk_1h,
        ml_risk_4h,
        ml_risk_8h,
        drug_masking,
        news2_score,
    ):
        # Simple entropy-only risk calculation
        risk = max(0.0, min(1.0, 1.0 - ces_adjusted))
        frs = int(round(risk * 100))

        # Apply drug masking bump
        if drug_masking and frs < 46:
            frs = min(frs + 10, 70)

        if frs <= 25:
            severity = "NONE"
        elif frs <= 45:
            severity = "WATCH"
        elif frs <= 70:
            severity = "WARNING"
        else:
            severity = "CRITICAL"

        return {
            "final_risk_score": frs,
            "final_severity": severity,
            "time_to_event_estimate": "Unknown (ML unavailable)",
            "component_risks": {"entropy": round(risk, 3)},
            "ml_available": False,
            "override_applied": None,
            "disagreement": None,
        }


class StubDetectorBank:
    """Stub for app.core.detectors.DetectorBank"""

    def run_all(self, **kwargs):
        """Returns list of 8 inactive detectors."""
        detector_names = [
            "entropy_threshold",
            "silent_decline",
            "drug_masking",
            "respiratory_risk",
            "hemodynamic",
            "alarm_suppression",
            "recovery",
            "data_quality",
        ]
        return [
            {
                "detector_name": name,
                "active": False,
                "severity": "NONE",
                "message": "",
                "contributing_factors": [],
                "recommended_action": "",
            }
            for name in detector_names
        ]
