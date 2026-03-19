"""
Data Generation Orchestrator.

Calls MIMIC-IV ingester + synthetic generator + feature engineer,
produces numpy arrays + metadata for training scripts.

Usage:
    python ml/generate_data.py [--mimic-dir data/mimic-iv-demo] [--output-dir data/ml]

Output:
    data/ml/X_train.npy, X_val.npy, X_test.npy
    data/ml/y_train_1h.npy, y_val_1h.npy, y_test_1h.npy  (+ 4h, 8h)
    data/ml/y_train_syndrome.npy, y_val_syndrome.npy, y_test_syndrome.npy
    data/ml/population_stats.json
    data/ml/generation_metadata.json
    data/ml/hero_cases/*.csv
"""

import sys
import os
import argparse
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.data.synthetic_generator import (
    MIMICIngester,
    create_statistical_clones,
    generate_trajectory_simulations,
    generate_deterioration_labels,
    select_hero_cases,
    save_hero_cases,
    PatientTrajectory,
)
from app.data.feature_engineer import FeatureEngineer


# Syndrome → integer mapping
SYNDROME_MAP = {
    "sepsis_like": 0,
    "respiratory_failure": 1,
    "hemodynamic_instability": 2,
    "cardiac_instability": 3,
    "stable": 4,
}


def trajectory_to_features(
    traj: PatientTrajectory,
    fe: FeatureEngineer,
    sample_minutes: list = None,
) -> tuple:
    """
    Extract feature vectors + labels from trajectory at given time points.

    Returns (X: list of (45,) arrays, labels_1h, labels_4h, labels_8h, syndrome_labels)
    """
    if sample_minutes is None:
        # Sample every 15 minutes from minute 30 onward
        sample_minutes = list(range(30, traj.duration_minutes - 1, 15))

    det_labels = generate_deterioration_labels(traj)

    X_list = []
    y_1h = []
    y_4h = []
    y_8h = []
    y_syn = []

    syndrome_int = SYNDROME_MAP.get(traj.label_syndrome, 4)

    for t in sample_minutes:
        if t >= traj.duration_minutes:
            continue

        # Build vitals window (up to this minute)
        window_start = max(0, t - 360)
        vitals_window = {
            "hr": traj.vitals["hr"][window_start:t + 1].tolist(),
            "bp_sys": traj.vitals["bp_sys"][window_start:t + 1].tolist(),
            "bp_dia": traj.vitals["bp_dia"][window_start:t + 1].tolist(),
            "rr": traj.vitals["rr"][window_start:t + 1].tolist(),
            "spo2": traj.vitals["spo2"][window_start:t + 1].tolist(),
            "temp": traj.vitals["temp"][window_start:t + 1].tolist(),
        }

        # Build mock entropy state from vitals
        hr_window = traj.vitals["hr"][window_start:t + 1]
        bp_window = traj.vitals["bp_sys"][window_start:t + 1]
        rr_window = traj.vitals["rr"][window_start:t + 1]
        spo2_window = traj.vitals["spo2"][window_start:t + 1]

        # Approximate SampEn using coefficient of variation (fast proxy)
        def approx_sampen(arr):
            if len(arr) < 10:
                return 1.5
            std = np.std(arr)
            mean = np.mean(arr)
            cv = std / max(abs(mean), 1.0)
            return max(0.01, min(2.5, cv * 10))

        ces_val = np.mean([
            approx_sampen(hr_window),
            approx_sampen(bp_window),
            approx_sampen(rr_window),
            approx_sampen(spo2_window),
        ]) / 2.5  # normalize to ~0-1

        entropy_state = {
            "sampen_hr": approx_sampen(hr_window),
            "sampen_bp_sys": approx_sampen(bp_window),
            "sampen_rr": approx_sampen(rr_window),
            "sampen_spo2": approx_sampen(spo2_window),
            "ces_adjusted": ces_val,
            "ces_raw": ces_val,
            "ces_slope_6h": 0.0,
            "window_size": min(t - window_start, 300),
        }

        # Build drug state at this minute
        active_drugs = []
        for d in traj.drugs:
            if d["minute"] <= t:
                active_drugs.append(d)

        drug_classes = set(d.get("drug_class", "") for d in active_drugs)
        drug_state = {
            "drug_masking": "vasopressor" in drug_classes or "sedative" in drug_classes,
            "active_drugs": active_drugs,
        }

        demographics = {"age": traj.age, "sex": traj.sex, "weight_kg": 75.0}

        # Compute feature vector
        features = fe.compute_features(vitals_window, entropy_state, drug_state, demographics)
        X_list.append(features)

        # Labels at this time point
        y_1h.append(det_labels["1h"][t])
        y_4h.append(det_labels["4h"][t])
        y_8h.append(det_labels["8h"][t])
        y_syn.append(syndrome_int)

    return X_list, y_1h, y_4h, y_8h, y_syn


def compute_population_stats(X: np.ndarray) -> dict:
    """Compute population-level statistics for warmup imputation."""
    stats = {}

    # Feature names for reference
    feature_names = [
        "age",
        "hr_current", "hr_mean_6h", "hr_min_6h", "hr_std_6h", "hr_trend_6h",
        "bp_sys_current", "bp_sys_mean_6h", "bp_sys_min_6h", "bp_sys_std_6h", "bp_sys_trend_6h",
        "rr_current", "rr_mean_6h", "rr_std_6h",
        "spo2_current", "spo2_mean_6h", "spo2_min_6h", "spo2_std_6h",
        "sampen_hr", "sampen_bp_sys", "sampen_rr", "sampen_spo2",
        "ces_adjusted", "ces_raw", "ces_slope_6h", "ces_velocity",
        "entropy_vital_divergence", "mse_slope_index",
        "shock_index", "map_current", "pulse_pressure", "rr_spo2_ratio",
        "vasopressor_active", "sedative_active", "beta_blocker_active", "opioid_active",
        "total_vasoactive_dose", "inotrope_active", "num_active_drugs",
        "temp_current", "temp_deviation",
        "window_fill_fraction", "time_since_drug_change",
        "hr_bp_correlation", "entropy_drug_interaction",
    ]

    stats["feature_names"] = feature_names

    # Compute medians for entropy features used in warmup imputation
    stats["sampen_hr_median"] = float(np.nanmedian(X[:, 18]))
    stats["sampen_bp_sys_median"] = float(np.nanmedian(X[:, 19]))
    stats["sampen_rr_median"] = float(np.nanmedian(X[:, 20]))
    stats["sampen_spo2_median"] = float(np.nanmedian(X[:, 21]))
    stats["ces_median"] = float(np.nanmedian(X[:, 22]))

    # Ranges for normalization
    for i, name in enumerate(feature_names):
        valid = X[:, i][~np.isnan(X[:, i])]
        if len(valid) > 0:
            stats[f"{name}_min"] = float(np.min(valid))
            stats[f"{name}_max"] = float(np.max(valid))
            stats[f"{name}_mean"] = float(np.mean(valid))
            stats[f"{name}_std"] = float(np.std(valid))

    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate ML training data")
    parser.add_argument("--mimic-dir", default="data/mimic-iv-demo")
    parser.add_argument("--output-dir", default="data/ml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clones-per-patient", type=int, default=5)
    parser.add_argument("--sim-per-template", type=int, default=40)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    t0 = time.time()

    # ═══════════════════════════════════════════════
    # Step 1: Ingest MIMIC-IV Demo (if available)
    # ═══════════════════════════════════════════════
    print("=" * 60)
    print("STEP 1: MIMIC-IV Demo Ingestion")
    print("=" * 60)
    ingester = MIMICIngester(args.mimic_dir)
    real_trajectories = ingester.ingest()
    print(f"  Real trajectories: {len(real_trajectories)}")

    # ═══════════════════════════════════════════════
    # Step 2: Statistical Clones
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 2: Statistical Cloning")
    print("=" * 60)
    if real_trajectories:
        clones = create_statistical_clones(
            real_trajectories,
            clones_per_patient=args.clones_per_patient,
            rng=rng,
        )
        print(f"  Clones created: {len(clones)}")
    else:
        clones = []
        print("  No real data — skipping cloning")

    # ═══════════════════════════════════════════════
    # Step 3: Trajectory Simulations
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 3: Trajectory Simulations")
    print("=" * 60)
    simulations = generate_trajectory_simulations(
        patients_per_template=args.sim_per_template,
        rng=rng,
    )
    print(f"  Simulations generated: {len(simulations)}")

    # Combine all trajectories
    all_trajectories = real_trajectories + clones + simulations
    print(f"\n  Total trajectories: {len(all_trajectories)}")

    # ═══════════════════════════════════════════════
    # Step 4: Feature Extraction
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 4: Feature Extraction (45-feature vectors)")
    print("=" * 60)

    fe = FeatureEngineer()

    all_X = []
    all_y1h = []
    all_y4h = []
    all_y8h = []
    all_ysyn = []

    # Track which trajectory each sample came from (for split)
    traj_ids = []

    for i, traj in enumerate(all_trajectories):
        if (i + 1) % 50 == 0:
            print(f"  Processing trajectory {i + 1}/{len(all_trajectories)}...")

        X_list, y1, y4, y8, ys = trajectory_to_features(traj, fe)
        all_X.extend(X_list)
        all_y1h.extend(y1)
        all_y4h.extend(y4)
        all_y8h.extend(y8)
        all_ysyn.extend(ys)
        traj_ids.extend([i] * len(X_list))

    X = np.array(all_X, dtype=np.float32)
    y_1h = np.array(all_y1h, dtype=np.int32)
    y_4h = np.array(all_y4h, dtype=np.int32)
    y_8h = np.array(all_y8h, dtype=np.int32)
    y_syn = np.array(all_ysyn, dtype=np.int32)
    traj_ids = np.array(traj_ids, dtype=np.int32)

    print(f"  Total samples: {X.shape[0]}, features: {X.shape[1]}")
    print(f"  NaN count: {np.isnan(X).sum()}")
    print(f"  Label balance (4h): {y_4h.sum()}/{len(y_4h)} positive ({y_4h.mean()*100:.1f}%)")

    # ═══════════════════════════════════════════════
    # Step 5: Train/Val/Test Split (by trajectory)
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 5: Train/Val/Test Split (trajectory-level)")
    print("=" * 60)

    unique_trajs = np.unique(traj_ids)
    rng.shuffle(unique_trajs)

    n_total = len(unique_trajs)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_trajs = set(unique_trajs[:n_train])
    val_trajs = set(unique_trajs[n_train:n_train + n_val])
    test_trajs = set(unique_trajs[n_train + n_val:])

    train_mask = np.array([t in train_trajs for t in traj_ids])
    val_mask = np.array([t in val_trajs for t in traj_ids])
    test_mask = np.array([t in test_trajs for t in traj_ids])

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train_1h, y_val_1h, y_test_1h = y_1h[train_mask], y_1h[val_mask], y_1h[test_mask]
    y_train_4h, y_val_4h, y_test_4h = y_4h[train_mask], y_4h[val_mask], y_4h[test_mask]
    y_train_8h, y_val_8h, y_test_8h = y_8h[train_mask], y_8h[val_mask], y_8h[test_mask]
    y_train_syn, y_val_syn, y_test_syn = y_syn[train_mask], y_syn[val_mask], y_syn[test_mask]

    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # ═══════════════════════════════════════════════
    # Step 6: Save Arrays
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 6: Saving to disk")
    print("=" * 60)

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "X_test.npy", X_test)

    for horizon in ["1h", "4h", "8h"]:
        y_tr = locals()[f"y_train_{horizon}"]
        y_va = locals()[f"y_val_{horizon}"]
        y_te = locals()[f"y_test_{horizon}"]
        np.save(output_dir / f"y_train_{horizon}.npy", y_tr)
        np.save(output_dir / f"y_val_{horizon}.npy", y_va)
        np.save(output_dir / f"y_test_{horizon}.npy", y_te)

    np.save(output_dir / "y_train_syndrome.npy", y_train_syn)
    np.save(output_dir / "y_val_syndrome.npy", y_val_syn)
    np.save(output_dir / "y_test_syndrome.npy", y_test_syn)

    # Population stats
    pop_stats = compute_population_stats(X_train)
    with open(output_dir / "population_stats.json", "w") as f:
        json.dump(pop_stats, f, indent=2)
    print(f"  Saved population_stats.json")

    # Generation metadata
    metadata = {
        "generated_at": datetime.utcnow().isoformat(),
        "seed": args.seed,
        "real_trajectories": len(real_trajectories),
        "statistical_clones": len(clones),
        "simulations": len(simulations),
        "total_trajectories": len(all_trajectories),
        "total_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "label_balance": {
            "1h_positive_rate": float(y_1h.mean()),
            "4h_positive_rate": float(y_4h.mean()),
            "8h_positive_rate": float(y_8h.mean()),
        },
        "syndrome_distribution": {k: int((y_syn == v).sum()) for k, v in SYNDROME_MAP.items()},
    }
    with open(output_dir / "generation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved generation_metadata.json")

    # ═══════════════════════════════════════════════
    # Step 7: Hero Cases
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 7: Hero Case Selection")
    print("=" * 60)

    heroes = select_hero_cases(simulations)
    save_hero_cases(heroes, str(output_dir / "hero_cases"))

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"COMPLETE in {elapsed:.1f}s")
    print(f"  Output directory: {output_dir}")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  Files saved: {len(list(output_dir.glob('*.npy')))} .npy + metadata")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
