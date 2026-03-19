"""
Evidence Engine Rebuilder.

Rebuilds the KNN tree using 45-feature vectors from the generated data,
replacing the existing 20-feature evidence tree.

Produces:
  data/models/historical_cases.parquet  (features + metadata)
  data/models/kdtree.joblib             (fitted KD-Tree for KNN queries)

Usage:
    python ml/rebuild_evidence.py [--data-dir data/ml] [--models-dir data/models]
"""

import sys
import argparse
import json
import time
import numpy as np
import joblib
from pathlib import Path

try:
    from scipy.spatial import KDTree
except ImportError:
    print("Missing scipy. Run: pip install scipy")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Rebuild evidence KNN tree")
    parser.add_argument("--data-dir", default="data/ml")
    parser.add_argument("--models-dir", default="data/models")
    parser.add_argument("--k-neighbors", type=int, default=15)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ─── Load training data ───
    print("=" * 60)
    print("Rebuilding Evidence KNN Tree")
    print("=" * 60)

    X_train = np.load(data_dir / "X_train.npy")
    y_train_syn = np.load(data_dir / "y_train_syndrome.npy")

    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Feature dimension: {X_train.shape[1]}")

    # Load scaler
    scaler = joblib.load(models_dir / "scaler.joblib")
    X_scaled = scaler.transform(X_train)

    # ─── Build KD-Tree ───
    print("\n  Building KD-Tree...")
    tree = KDTree(X_scaled)

    # ─── Save ───
    joblib.dump(tree, models_dir / "kdtree.joblib")
    print(f"  Saved kdtree.joblib")

    # Save a compressed version of the feature matrix + labels for lookup
    np.savez_compressed(
        models_dir / "evidence_data.npz",
        features=X_scaled,
        syndrome_labels=y_train_syn,
    )
    print(f"  Saved evidence_data.npz")

    # ─── Verify ───
    print("\n  Verification: querying random point...")
    test_point = X_scaled[0:1]
    distances, indices = tree.query(test_point, k=min(5, len(X_scaled)))
    print(f"  Nearest 5 distances: {distances[0][:5]}")
    print(f"  Nearest 5 indices: {indices[0][:5]}")

    # Stats
    stats = {
        "n_cases": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "k_neighbors": args.k_neighbors,
        "tree_type": "KDTree",
    }
    with open(models_dir / "evidence_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved evidence_stats.json")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"EVIDENCE REBUILD COMPLETE in {elapsed:.1f}s")
    print(f"  Cases: {X_train.shape[0]}")
    print(f"  Dimensions: {X_train.shape[1]}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
