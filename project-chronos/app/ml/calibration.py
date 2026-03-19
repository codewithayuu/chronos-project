"""
Calibrated model wrapper for sklearn 1.8+ compatibility.

sklearn 1.8 removed cv='prefit' from CalibratedClassifierCV.
This module provides CalibratedModel — a Platt sigmoid wrapper
that is pickle-safe (importable from a persistent location).

Also provides TemperatureScaler for multi-class calibration.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class CalibratedModel:
    """
    Wraps a trained model with Platt sigmoid calibration.

    Responds to predict_proba(X) → (N, 2) exactly like
    CalibratedClassifierCV did.
    """

    def __init__(self, base_model, X_val_scaled=None, y_val=None):
        self.base_model = base_model
        self.calibrator = None
        if X_val_scaled is not None and y_val is not None:
            raw_probs = base_model.predict_proba(X_val_scaled)[:, 1].reshape(-1, 1)
            self.calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            self.calibrator.fit(raw_probs, y_val)

    def predict_proba(self, X):
        """Return calibrated (N, 2) probability matrix."""
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        if self.calibrator is not None:
            cal_pos = self.calibrator.predict_proba(raw)[:, 1]
        else:
            cal_pos = raw.ravel()
        return np.column_stack([1 - cal_pos, cal_pos])

    def predict(self, X):
        """Return binary predictions."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    @property
    def calibrated_classifiers_(self):
        """Compatibility shim so predictor.py can find the XGBoost booster."""
        class _Stub:
            def __init__(self, est):
                self.estimator = est
        return [_Stub(self.base_model)]


class TemperatureScaler:
    """
    Temperature scaling for calibrated multi-class probabilities.

    Learns a single temperature parameter T that divides logits
    before softmax to improve calibration.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        """Fit temperature parameter on validation set."""
        from scipy.optimize import minimize_scalar

        def nll(T):
            T = max(T, 0.01)
            scaled = logits / T
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            eps = 1e-12
            return -np.mean(np.log(probs[np.arange(len(labels)), labels] + eps))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = result.x
        print(f"  Temperature scaling: T = {self.temperature:.4f}")
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to probability matrix."""
        eps = 1e-12
        logits = np.log(probs + eps)
        scaled = logits / self.temperature
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
