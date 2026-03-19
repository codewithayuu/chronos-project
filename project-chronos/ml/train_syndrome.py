
"""
Syndrome Classifier Trainer.

Trains an XGBoost multi-class classifier for 5 syndrome types,
with temperature scaling for calibrated probabilities.

Syndrome classes:
  0 = sepsis_like
  1 = respiratory_failure
  2 = hemodynamic_instability
  3 = cardiac_instability
  4 = stable

Usage:
    python ml/train_syndrome.py [--data-dir data/ml] [--models-dir data/models]
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

try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install xgboost scikit-learn scipy joblib")
    sys.exit(1)

from app.ml.calibration import TemperatureScaler


SYNDROME_NAMES = [
    "sepsis_like",
    "respiratory_failure",
    "hemodynamic_instability",
    "cardiac_instability",
    "stable",
]


def main():
    parser = argparse.ArgumentParser(description="Train syndrome classifier")
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
    y_train = np.load(data_dir / "y_train_syndrome.npy")
    y_val = np.load(data_dir / "y_val_syndrome.npy")
    y_test = np.load(data_dir / "y_test_syndrome.npy")

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  Class distribution (train):")
    for i, name in enumerate(SYNDROME_NAMES):
        count = (y_train == i).sum()
        print(f"    {name}: {count} ({count / len(y_train) * 100:.1f}%)")

    # Load scaler (from deterioration training)
    scaler = joblib.load(models_dir / "scaler.joblib")
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ─── Train classifier ───
    print("\n" + "=" * 60)
    print("Training Syndrome Classifier")
    print("=" * 60)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
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

    joblib.dump(model, models_dir / "syndrome_classifier.joblib")
    print(f"  Saved syndrome_classifier.joblib")

    # ─── Temperature scaling ───
    print("\n  Calibrating with temperature scaling...")
    raw_probs_val = model.predict_proba(X_val_scaled)

    # Get raw logits for temperature fitting
    # XGBoost raw output → we use log(probs) as proxy logits
    eps = 1e-12
    logits_val = np.log(raw_probs_val + eps)

    temp_scaler = TemperatureScaler()
    temp_scaler.fit(logits_val, y_val)

    joblib.dump(temp_scaler, models_dir / "temperature_scaler.joblib")
    print(f"  Saved temperature_scaler.joblib")

    # ─── Evaluation ───
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

    # Validation
    raw_probs_val = model.predict_proba(X_val_scaled)
    cal_probs_val = temp_scaler.calibrate(raw_probs_val)
    y_pred_val = np.argmax(cal_probs_val, axis=1)
    acc_val = accuracy_score(y_val, y_pred_val)
    print(f"\n  Validation Accuracy: {acc_val:.4f}")

    # Test
    raw_probs_test = model.predict_proba(X_test_scaled)
    cal_probs_test = temp_scaler.calibrate(raw_probs_test)
    y_pred_test = np.argmax(cal_probs_test, axis=1)
    acc_test = accuracy_score(y_test, y_pred_test)
    print(f"  Test Accuracy: {acc_test:.4f}")

    print(f"\n  Classification Report (Test):")
    report = classification_report(
        y_test,
        y_pred_test,
        target_names=SYNDROME_NAMES,
        zero_division=0,
    )
    print(report)

    # ─── Save metrics ───
    metrics = {
        "val_accuracy": round(acc_val, 4),
        "test_accuracy": round(acc_test, 4),
        "temperature": round(temp_scaler.temperature, 4),
        "n_classes": 5,
        "syndrome_names": SYNDROME_NAMES,
        "train_distribution": {
            name: int((y_train == i).sum()) for i, name in enumerate(SYNDROME_NAMES)
        },
    }
    with open(models_dir / "syndrome_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved syndrome_metrics.json")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"SYNDROME TRAINING COMPLETE in {elapsed:.1f}s")
    print(f"  Val accuracy: {acc_val:.4f}")
    print(f"  Test accuracy: {acc_test:.4f}")
    print(f"  Temperature: {temp_scaler.temperature:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
