"""
Syndrome Classifier — ML Runtime for Project Chronos.

Wraps the XGBoost multi-class syndrome classifier with:
  - Temperature-scaled calibration for honest confidence scores
  - 5-class syndrome mapping (sepsis, respiratory, hemodynamic, cardiac, stable)
  - Inconclusive detection when confidence is below threshold
  - Graceful degradation when model files are missing
"""

import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

import joblib


class SyndromeClassifier:
    """
    Wraps the XGBoost syndrome classifier with temperature scaling.

    5 syndrome classes:
      0: sepsis_like
      1: respiratory_failure
      2: hemodynamic_instability
      3: cardiac_instability
      4: stable
    """

    SYNDROME_NAMES = [
        "sepsis_like",
        "respiratory_failure",
        "hemodynamic_instability",
        "cardiac_instability",
        "stable",
    ]
    SYNDROME_DISPLAY = [
        "Sepsis-like",
        "Respiratory Failure",
        "Hemodynamic Instability",
        "Cardiac Instability",
        "Stable",
    ]
    MIN_CONFIDENCE = 0.40

    def __init__(self, models_dir: str = "data/models"):
        self.available: bool = False
        self.models_dir = Path(models_dir)
        self.model: Any = None
        self.temp_scaler: Any = None
        self.scaler: Any = None

        try:
            self.model = joblib.load(self.models_dir / "syndrome_classifier.joblib")
            self.temp_scaler = joblib.load(self.models_dir / "temperature_scaler.joblib")
            self.scaler = joblib.load(self.models_dir / "scaler.joblib")
            self.available = True
            self._validate_loaded_models()
            print("[SyndromeClassifier] Syndrome classifier loaded successfully.")
        except FileNotFoundError as e:
            print(f"[SyndromeClassifier] WARNING: Model file not found: {e}. Classifier unavailable.")
        except Exception as e:
            print(f"[SyndromeClassifier] WARNING: Failed to load models: {e}. Classifier unavailable.")

    def reload(self) -> bool:
        """
        Hot-reload model from disk when Person 1 delivers new files.
        Returns True if successful.
        """
        old_available = self.available
        self.available = False
        try:
            self.model = joblib.load(self.models_dir / "syndrome_classifier.joblib")
            self.temp_scaler = joblib.load(self.models_dir / "temperature_scaler.joblib")
            self.scaler = joblib.load(self.models_dir / "scaler.joblib")
            self.available = True
            self._validate_loaded_models()
            print("[SyndromeClassifier] Models reloaded successfully.")
            return True
        except Exception as e:
            self.available = old_available
            print(f"[SyndromeClassifier] Reload failed: {e}")
            return False

    def _validate_loaded_models(self):
        """Runtime validation of loaded models."""
        # Check scaler dimensions
        if hasattr(self.scaler, 'n_features_in_'):
            if self.scaler.n_features_in_ != 45:
                print(f"[SyndromeClassifier] WARNING: Scaler expects "
                      f"{self.scaler.n_features_in_} features, runtime expects 45")

        # Check class count
        if hasattr(self.model, 'n_classes_'):
            if self.model.n_classes_ != 5:
                print(f"[SyndromeClassifier] WARNING: Model has "
                      f"{self.model.n_classes_} classes, expected 5")
        elif hasattr(self.model, 'classes_'):
            if len(self.model.classes_) != 5:
                print(f"[SyndromeClassifier] WARNING: Model has "
                      f"{len(self.model.classes_)} classes, expected 5")

        # Check temperature scaler interface
        if not hasattr(self.temp_scaler, 'calibrate'):
            print("[SyndromeClassifier] WARNING: Temperature scaler "
                  "missing .calibrate() method")

    def predict(self, features: np.ndarray) -> Optional[dict]:
        """
        Classify the patient's deterioration syndrome pattern.

        Parameters
        ----------
        features : np.ndarray, shape (45,)
            Raw 45-element feature vector from FeatureEngineer.

        Returns
        -------
        dict or None
            Syndrome classification with calibrated confidence, or None if unavailable.
        """
        if not self.available:
            return None

        try:
            # Validate input
            features = np.asarray(features, dtype=np.float64).flatten()
            if features.shape[0] != 45:
                print(f"[SyndromeClassifier] ERROR: Expected 45 features, got {features.shape[0]}")
                return None

            # Replace any NaN/inf with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale features
            X = self.scaler.transform(features.reshape(1, -1))

            # Get raw probabilities
            raw_probs = self.model.predict_proba(X)  # shape (1, 5)

            # Apply temperature scaling for calibration
            calibrated = self.temp_scaler.calibrate(raw_probs)[0]  # shape (5,)

            # Ensure probabilities still sum to 1 after calibration
            calibrated = np.clip(calibrated, 0.0, 1.0)
            prob_sum = calibrated.sum()
            if prob_sum > 0:
                calibrated = calibrated / prob_sum

            # Sort by probability descending
            sorted_indices = np.argsort(calibrated)[::-1]
            primary_idx = sorted_indices[0]
            secondary_idx = sorted_indices[1]

            primary_conf = float(calibrated[primary_idx])
            secondary_conf = float(calibrated[secondary_idx])

            return {
                "primary_syndrome": self.SYNDROME_DISPLAY[primary_idx],
                "primary_confidence": float(f"{primary_conf:.2f}"),
                "secondary_syndrome": (
                    self.SYNDROME_DISPLAY[secondary_idx]
                    if secondary_conf > 0.20
                    else None
                ),
                "secondary_confidence": (
                    float(f"{secondary_conf:.2f}") if secondary_conf > 0.20 else None
                ),
                "all_probabilities": {
                    name: float(f"{float(calibrated[i]):.3f}")
                    for i, name in enumerate(self.SYNDROME_NAMES)
                },
                "inconclusive": primary_conf < self.MIN_CONFIDENCE,
                "disclaimer": "Pattern similarity assessment, not a clinical diagnosis",
            }
        except Exception as e:
            print(f"[SyndromeClassifier] ERROR during prediction: {e}")
            return None
