"""
Deterioration Model Trainer.

Trains 3 XGBoost classifiers (1h, 4h, 8h horizons) with Platt sigmoid calibration.
Outputs calibrated model wrappers, scaler, and confidence params.

Usage:
    python ml/train_deterioration.py [--data-dir data/ml] [--models-dir data/models]
"""

import sys
import argparse
import json
import time
import numpy as np
import joblib
from pathlib import Path

# Add project root to sys.path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# XGBoost + sklearn
try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install xgboost scikit-learn joblib")
    sys.exit(1)

from app.ml.calibration import CalibratedModel


def train_horizon(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    horizon_name: str,
    scaler: StandardScaler,
) -> tuple:
    """
    Train one XGBoost model for a single horizon, calibrate on val set.

    Returns (calibrated_model, metrics_dict)
    """
    print(f"\n  --- Training {horizon_name} model ---")

    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Compute pos_weight for THIS horizon (Bug Fix #1 from PRD)
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = n_neg / max(n_pos, 1)
    print(f"  Pos weight: {pos_weight:.2f} (pos={n_pos}, neg={n_neg})")

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False,
    )

    # Calibrate using Platt sigmoid scaling on val set
    calibrated = CalibratedModel(model, X_val_scaled, y_val)

    # Evaluate
    y_pred_proba = calibrated.predict_proba(X_val_scaled)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
    ap = average_precision_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.0

    metrics = {
        "horizon": horizon_name,
        "auc_roc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "pos_rate_train": round(float(y_train.mean()), 4),
        "pos_rate_val": round(float(y_val.mean()), 4),
    }
    print(f"  AUC-ROC: {auc:.4f}, Avg Precision: {ap:.4f}")

    return calibrated, metrics


def compute_confidence_params(X_train_scaled: np.ndarray) -> dict:
    """
    Compute Mahalanobis distance parameters for confidence estimation.
    Uses pseudo-inverse for stability with high-dimensional features.
    """
    centroid = np.mean(X_train_scaled, axis=0)
    cov = np.cov(X_train_scaled, rowvar=False)
    # Use pseudo-inverse for stability (45 features with ~N samples)
    cov_inv = np.linalg.pinv(cov)

    return {
        "centroid": centroid,
        "cov_inv": cov_inv,
    }


def main():
    parser = argparse.ArgumentParser(description="Train deterioration models")
    parser.add_argument("--data-dir", default="data/ml")
    parser.add_argument("--models-dir", default="data/models")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ─── Load data ───
    print("=" * 60)
    print("Loading training data...")
    print("=" * 60)

    X_train = np.load(data_dir / "X_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    X_test = np.load(data_dir / "X_test.npy")

    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    # ─── Fit scaler ───
    print("\nFitting StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, models_dir / "scaler.joblib")
    print(f"  Saved scaler.joblib")

    # ─── Train for each horizon ───
    print("\n" + "=" * 60)
    print("Training Deterioration Models")
    print("=" * 60)

    all_metrics = {}
    for horizon in ["1h", "4h", "8h"]:
        y_train = np.load(data_dir / f"y_train_{horizon}.npy")
        y_val = np.load(data_dir / f"y_val_{horizon}.npy")

        model, metrics = train_horizon(
            X_train, y_train, X_val, y_val, horizon, scaler
        )

        model_path = models_dir / f"deterioration_{horizon}.joblib"
        joblib.dump(model, model_path)
        print(f"  Saved {model_path}")

        all_metrics[horizon] = metrics

    # ─── Test set evaluation ───
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    X_test_scaled = scaler.transform(X_test)
    for horizon in ["1h", "4h", "8h"]:
        y_test = np.load(data_dir / f"y_test_{horizon}.npy")
        model = joblib.load(models_dir / f"deterioration_{horizon}.joblib")
        y_pred = model.predict_proba(X_test_scaled)[:, 1]

        auc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.5
        print(f"  {horizon}: AUC-ROC = {auc:.4f} (n={len(y_test)})")
        all_metrics[horizon]["test_auc_roc"] = round(auc, 4)

    # ─── Confidence parameters ───
    print("\nComputing confidence parameters...")
    X_train_scaled = scaler.transform(X_train)
    conf_params = compute_confidence_params(X_train_scaled)
    joblib.dump(conf_params, models_dir / "confidence_params.joblib")
    print(f"  Saved confidence_params.joblib")
    print(f"  Centroid shape: {conf_params['centroid'].shape}")
    print(f"  Cov_inv shape: {conf_params['cov_inv'].shape}")

    # ─── Save metrics ───
    with open(models_dir / "deterioration_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved deterioration_metrics.json")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"DETERIORATION TRAINING COMPLETE in {elapsed:.1f}s")
    print(f"  Models: {models_dir}")
    for h, m in all_metrics.items():
        print(f"    {h}: AUC={m['auc_roc']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
