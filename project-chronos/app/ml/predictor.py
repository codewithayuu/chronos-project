"""
Deterioration Predictor — ML Runtime for Project Chronos.

Wraps three XGBoost models (1h, 4h, 8h risk horizons) with:
  - Graceful degradation when model files are missing
  - Per-prediction feature importance via XGBoost pred_contribs
  - Mahalanobis-distance-based confidence scoring
  - StandardScaler integration for feature normalization
"""

import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path

import joblib
import xgboost as xgb


# ──────────────────────────────────────────────
# 45-Feature Name Registry
# ──────────────────────────────────────────────
FEATURE_NAMES = [
    # Current vitals (0-5)
    "hr_current", "bp_sys_current", "bp_dia_current",
    "rr_current", "spo2_current", "temp_current",
    # Vital statistics over 6h window (6-17)
    "hr_mean_6h", "hr_std_6h", "bp_sys_mean_6h", "bp_sys_std_6h",
    "rr_mean_6h", "rr_std_6h", "spo2_mean_6h", "spo2_std_6h",
    "temp_mean_6h", "temp_std_6h", "hr_min_6h", "hr_max_6h",
    # Entropy features (18-27)
    "sampen_hr", "sampen_bp_sys", "sampen_rr", "sampen_spo2",
    "ces_raw", "ces_adjusted", "ces_slope_6h", "ces_slope_1h",
    "mse_slope_index", "entropy_vital_divergence",
    # Derived clinical indices (28-33)
    "shock_index", "modified_shock_index", "pulse_pressure",
    "map_value", "rr_spo2_ratio", "hr_rr_ratio",
    # Drug context (34-37)
    "vasopressor_active", "inotrope_active",
    "total_vasoactive_dose", "num_active_drugs",
    # Trend features (38-41)
    "hr_trend_slope", "bp_sys_trend_slope",
    "rr_trend_slope", "spo2_trend_slope",
    # Composite / interaction features (42-44)
    "entropy_vital_divergence_x_ces", "news2_score", "age_normalized",
]

FEATURE_DESCRIPTIONS = {
    "hr_current": "Current heart rate",
    "bp_sys_current": "Current systolic blood pressure",
    "bp_dia_current": "Current diastolic blood pressure",
    "rr_current": "Current respiratory rate",
    "spo2_current": "Current oxygen saturation",
    "temp_current": "Current temperature",
    "hr_mean_6h": "Average heart rate (6h)",
    "hr_std_6h": "Heart rate variability (6h)",
    "bp_sys_mean_6h": "Average systolic BP (6h)",
    "bp_sys_std_6h": "Blood pressure variability (6h)",
    "rr_mean_6h": "Average respiratory rate (6h)",
    "rr_std_6h": "Respiratory rate variability (6h)",
    "spo2_mean_6h": "Average oxygen saturation (6h)",
    "spo2_std_6h": "Unstable oxygen saturation",
    "temp_mean_6h": "Average temperature (6h)",
    "temp_std_6h": "Temperature variability (6h)",
    "hr_min_6h": "Minimum heart rate (6h)",
    "hr_max_6h": "Maximum heart rate (6h)",
    "sampen_hr": "Heart rate complexity",
    "sampen_bp_sys": "Blood pressure complexity",
    "sampen_rr": "Low respiratory complexity",
    "sampen_spo2": "Oxygen saturation complexity",
    "ces_raw": "Raw composite entropy",
    "ces_adjusted": "Drug-adjusted composite entropy",
    "ces_slope_6h": "Declining entropy trend",
    "ces_slope_1h": "Short-term entropy trend",
    "mse_slope_index": "Multi-scale entropy slope",
    "entropy_vital_divergence": "Entropy divergence across vitals",
    "shock_index": "Elevated shock index",
    "modified_shock_index": "Modified shock index",
    "pulse_pressure": "Pulse pressure",
    "map_value": "Mean arterial pressure",
    "rr_spo2_ratio": "Respiratory-oxygenation ratio",
    "hr_rr_ratio": "Heart-respiratory coupling",
    "vasopressor_active": "Vasopressor dependency",
    "inotrope_active": "Inotrope dependency",
    "total_vasoactive_dose": "Total vasopressor load",
    "num_active_drugs": "Number of active medications",
    "hr_trend_slope": "Heart rate trend",
    "bp_sys_trend_slope": "Blood pressure trend",
    "rr_trend_slope": "Respiratory rate trend",
    "spo2_trend_slope": "Oxygen saturation trend",
    "entropy_vital_divergence_x_ces": "Divergence-entropy interaction",
    "news2_score": "NEWS2 clinical score",
    "age_normalized": "Age factor",
}


class DeteriorationPredictor:
    """
    Wraps the three XGBoost deterioration risk models (1h, 4h, 8h).

    - Loads CalibratedClassifierCV-wrapped XGBoost models from disk.
    - Computes Mahalanobis-based confidence for each prediction.
    - Extracts per-prediction feature contributions (Bug Fix #3).
    - Gracefully returns None when model files are unavailable.
    """

    CONFIDENCE_HIGH_THRESHOLD = 2.0
    CONFIDENCE_MODERATE_THRESHOLD = 3.5

    def __init__(self, models_dir: str = "data/models"):
        self.available: bool = False
        self.models_dir = Path(models_dir)
        self.model_1h: Any = None
        self.model_4h: Any = None
        self.model_8h: Any = None
        self.scaler: Any = None
        self.centroid: Any = None
        self.cov_inv: Any = None

        try:
            self.model_1h = joblib.load(self.models_dir / "deterioration_1h.joblib")
            self.model_4h = joblib.load(self.models_dir / "deterioration_4h.joblib")
            self.model_8h = joblib.load(self.models_dir / "deterioration_8h.joblib")
            self.scaler = joblib.load(self.models_dir / "scaler.joblib")
            conf = joblib.load(self.models_dir / "confidence_params.joblib")
            self.centroid = conf["centroid"]
            self.cov_inv = conf["cov_inv"]
            self.available = True
            self._validate_loaded_models()
            print("[DeteriorationPredictor] All 3 horizon models loaded successfully.")
        except FileNotFoundError as e:
            print(f"[DeteriorationPredictor] WARNING: Model file not found: {e}. Predictor unavailable.")
        except Exception as e:
            print(f"[DeteriorationPredictor] WARNING: Failed to load models: {e}. Predictor unavailable.")

    def reload(self) -> bool:
        """
        Hot-reload models from disk. Useful when Person 1 delivers new model files
        during integration without restarting the server.

        Returns True if models were loaded successfully.
        """
        old_available = self.available
        self.available = False
        try:
            self.model_1h = joblib.load(self.models_dir / "deterioration_1h.joblib")
            self.model_4h = joblib.load(self.models_dir / "deterioration_4h.joblib")
            self.model_8h = joblib.load(self.models_dir / "deterioration_8h.joblib")
            self.scaler = joblib.load(self.models_dir / "scaler.joblib")
            conf = joblib.load(self.models_dir / "confidence_params.joblib")
            self.centroid = conf["centroid"]
            self.cov_inv = conf["cov_inv"]
            self.available = True
            self._validate_loaded_models()
            print("[DeteriorationPredictor] Models reloaded successfully.")
            return True
        except Exception as e:
            self.available = old_available
            print(f"[DeteriorationPredictor] Reload failed: {e}")
            return False

    def _validate_loaded_models(self):
        """Runtime validation of loaded models for compatibility."""
        # Check scaler dimensions
        if hasattr(self.scaler, 'n_features_in_'):
            if self.scaler.n_features_in_ != 45:
                print(f"[DeteriorationPredictor] WARNING: Scaler expects "
                      f"{self.scaler.n_features_in_} features, runtime expects 45")

        # Check confidence param shapes
        if self.centroid is not None and self.centroid.shape != (45,):
            print(f"[DeteriorationPredictor] WARNING: Centroid shape "
                  f"{self.centroid.shape}, expected (45,)")
        if self.cov_inv is not None and self.cov_inv.shape != (45, 45):
            print(f"[DeteriorationPredictor] WARNING: Covariance inverse shape "
                  f"{self.cov_inv.shape}, expected (45, 45)")

    def predict(self, features: np.ndarray) -> Optional[dict]:
        """
        Predict deterioration risk at 1h, 4h, 8h horizons.

        Parameters
        ----------
        features : np.ndarray, shape (45,)
            Raw 45-element feature vector from FeatureEngineer.

        Returns
        -------
        dict or None
            Risk predictions with confidence and top drivers, or None if unavailable.
        """
        if not self.available:
            return None

        try:
            # Validate input
            features = np.asarray(features, dtype=np.float64).flatten()
            if features.shape[0] != 45:
                print(f"[DeteriorationPredictor] ERROR: Expected 45 features, got {features.shape[0]}")
                return None

            # Replace any NaN/inf with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Predict risk at each horizon
            risk_1h = float(self.model_1h.predict_proba(features_scaled)[0, 1])
            risk_4h = float(self.model_4h.predict_proba(features_scaled)[0, 1])
            risk_8h = float(self.model_8h.predict_proba(features_scaled)[0, 1])

            # Compute confidence via Mahalanobis distance
            confidence = self._compute_confidence(features_scaled[0])

            # Extract per-prediction feature drivers (using 4h model as primary)
            top_drivers = self._extract_drivers(features_scaled, self.model_4h, top_k=5)

            return {
                "risk_1h": float(f"{risk_1h:.4f}"),
                "risk_4h": float(f"{risk_4h:.4f}"),
                "risk_8h": float(f"{risk_8h:.4f}"),
                "model_confidence": confidence,
                "top_drivers": top_drivers,
            }
        except Exception as e:
            print(f"[DeteriorationPredictor] ERROR during prediction: {e}")
            return None

    def _compute_confidence(self, features_scaled: np.ndarray) -> str:
        """
        Compute prediction confidence using Mahalanobis distance
        from the training distribution centroid.

        Low distance = high confidence (the patient is similar to training data).
        """
        if self.centroid is None or self.cov_inv is None:
            return "moderate"

        try:
            diff = features_scaled - self.centroid
            mahal_dist = float(np.sqrt(diff @ self.cov_inv @ diff))

            if mahal_dist <= self.CONFIDENCE_HIGH_THRESHOLD:
                return "high"
            elif mahal_dist <= self.CONFIDENCE_MODERATE_THRESHOLD:
                return "moderate"
            else:
                return "low"
        except Exception:
            return "moderate"

    def _extract_drivers(
        self, features_scaled: np.ndarray, model, top_k: int = 5
    ) -> List[Dict]:
        """
        Per-prediction feature contributions using XGBoost's built-in
        pred_contribs method (Bug Fix #3).

        This gives patient-specific explanations instead of global importance.
        """
        if xgb is None:
            return self._extract_drivers_fallback(model, top_k)

        try:
            # Navigate through CalibratedClassifierCV to get the XGBoost booster
            if hasattr(model, "calibrated_classifiers_"):
                estimator = model.calibrated_classifiers_[0].estimator
                booster = estimator.get_booster()
            elif hasattr(model, "get_booster"):
                booster = model.get_booster()
            else:
                return self._extract_drivers_fallback(model, top_k)

            dmatrix = xgb.DMatrix(
                features_scaled.reshape(1, -1), feature_names=FEATURE_NAMES
            )

            # pred_contribs returns (1, 46) — 45 feature contributions + 1 bias term
            contribs = booster.predict(dmatrix, pred_contribs=True)[0]

            # Remove bias term (last element)
            feature_contribs = contribs[:-1]  # shape (45,)

            # Take absolute values (we care about magnitude)
            abs_contribs = np.abs(feature_contribs)

            # Get top-k indices
            top_indices = np.argsort(abs_contribs)[::-1][:top_k]

            total = abs_contribs[top_indices].sum()

            drivers = []
            for idx in top_indices:
                name = FEATURE_NAMES[idx]
                drivers.append(
                    {
                        "feature": name,
                        "description": FEATURE_DESCRIPTIONS.get(name, name),
                        "importance": float(f"{abs_contribs[idx] / total:.3f}") if total > 0 else 0.0,
                        "direction": "increases_risk"
                        if feature_contribs[idx] > 0
                        else "decreases_risk",
                    }
                )

            return drivers

        except Exception as e:
            print(f"[DeteriorationPredictor] pred_contribs failed: {e}, using fallback.")
            return self._extract_drivers_fallback(model, top_k)

    def _extract_drivers_fallback(self, model, top_k: int = 5) -> List[Dict]:
        """
        Fallback: use global feature importance when pred_contribs is unavailable.
        """
        try:
            if hasattr(model, "calibrated_classifiers_"):
                estimator = model.calibrated_classifiers_[0].estimator
            elif hasattr(model, "feature_importances_"):
                estimator = model
            else:
                return []

            importances = estimator.feature_importances_
            top_indices = np.argsort(importances)[::-1][:top_k]
            total = importances[top_indices].sum()

            drivers = []
            for idx in top_indices:
                name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
                drivers.append(
                    {
                        "feature": name,
                        "description": FEATURE_DESCRIPTIONS.get(name, name),
                        "importance": float(f"{importances[idx] / total:.3f}") if total > 0 else 0.0,
                        "direction": "increases_risk",  # cannot determine per-sample
                    }
                )

            return drivers
        except Exception:
            return []
