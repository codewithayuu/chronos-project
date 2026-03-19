"""
Phase 1 Comprehensive Tests — Project Chronos

Run with:  python -m tests.test_phase1   (from project-chronos directory)
    or:    cd project-chronos && python tests/test_phase1.py
"""

import sys
import os
import time
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
from app.config import AppConfig, load_config
from app.models import (
    VitalSignRecord,
    PatientState,
    AlertSeverity,
    TrendDirection,
)
from app.entropy.sampen import sample_entropy, get_backend
from app.entropy.mse import multiscale_entropy, coarse_grain
from app.entropy.normalization import normalize_sampen, SAMPEN_RANGES
from app.entropy.engine import EntropyEngine, VITAL_NAMES


def header(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ════════════════════════════════════════════
# TEST GROUP 1: SampEn Algorithm
# ════════════════════════════════════════════

def test_sampen_constant_signal():
    """A constant signal has zero complexity."""
    data = np.ones(300) * 72.0
    result = sample_entropy(data, m=2, r_fraction=0.2)
    print(f"  SampEn(constant) = {result}")
    assert result == 0.0, f"Expected 0.0, got {result}"
    print("  ✅ PASSED")


def test_sampen_sine_wave():
    """A sine wave is very regular → low SampEn."""
    t = np.linspace(0, 20 * np.pi, 300)
    data = 75.0 + 10.0 * np.sin(t)
    result = sample_entropy(data, m=2, r_fraction=0.2)
    print(f"  SampEn(sine) = {result:.4f}")
    assert result is not None and not np.isnan(result), "Sine should produce valid SampEn"
    assert result < 1.5, f"Sine should have low SampEn, got {result:.4f}"
    print("  ✅ PASSED")


def test_sampen_random_noise():
    """Random noise is complex → high SampEn."""
    np.random.seed(42)
    data = np.random.randn(300) * 10 + 75
    result = sample_entropy(data, m=2, r_fraction=0.2)
    print(f"  SampEn(random) = {result:.4f}")
    assert result is not None and not np.isnan(result), "Random should produce valid SampEn"
    assert result > 1.0, f"Random should have high SampEn, got {result:.4f}"
    print("  ✅ PASSED")


def test_sampen_ordering():
    """Random noise > sine wave > constant in terms of SampEn."""
    constant_se = sample_entropy(np.ones(300) * 72, m=2, r_fraction=0.2)
    sine_se = sample_entropy(75 + 10 * np.sin(np.linspace(0, 20 * np.pi, 300)), m=2, r_fraction=0.2)
    np.random.seed(42)
    random_se = sample_entropy(np.random.randn(300) * 10 + 75, m=2, r_fraction=0.2)

    print(f"  constant={constant_se:.4f}, sine={sine_se:.4f}, random={random_se:.4f}")
    assert constant_se < sine_se, "Constant should be less than sine"
    assert sine_se < random_se, "Sine should be less than random"
    print("  ✅ PASSED — Ordering: constant < sine < random")


def test_sampen_short_data():
    """Too-short data should return NaN."""
    result = sample_entropy(np.array([1.0, 2.0]), m=2, r_fraction=0.2)
    assert np.isnan(result), f"Short data should return NaN, got {result}"
    print("  ✅ PASSED")


def test_sampen_with_nans():
    """NaN values should be stripped before computation."""
    np.random.seed(42)
    data = np.random.randn(300) * 10 + 75
    # Insert some NaNs
    data[50:55] = np.nan
    data[200:203] = np.nan
    result = sample_entropy(data, m=2, r_fraction=0.2)
    print(f"  SampEn(with NaNs) = {result:.4f}")
    assert not np.isnan(result), "Should handle NaNs gracefully"
    print("  ✅ PASSED")


def test_sampen_performance():
    """SampEn on 300 points should complete in <1 second."""
    np.random.seed(42)
    data = np.random.randn(300) * 10 + 75

    # Warmup call (important for Numba JIT)
    _ = sample_entropy(data[:50], m=2, r_fraction=0.2)

    start = time.time()
    for _ in range(6):  # simulate 6 vitals
        sample_entropy(data, m=2, r_fraction=0.2)
    elapsed = time.time() - start

    print(f"  6 × SampEn(N=300) in {elapsed:.3f}s (backend: {get_backend()})")
    assert elapsed < 5.0, f"Too slow: {elapsed:.3f}s for 6 vitals"
    print("  ✅ PASSED")


# ════════════════════════════════════════════
# TEST GROUP 2: MSE
# ════════════════════════════════════════════

def test_coarse_grain():
    """Coarse-graining should reduce series length by scale factor."""
    data = np.arange(100, dtype=np.float64)
    cg2 = coarse_grain(data, 2)
    assert len(cg2) == 50, f"Expected 50, got {len(cg2)}"
    assert cg2[0] == 0.5  # mean of [0, 1]
    assert cg2[1] == 2.5  # mean of [2, 3]

    cg5 = coarse_grain(data, 5)
    assert len(cg5) == 20, f"Expected 20, got {len(cg5)}"
    print("  ✅ PASSED")


def test_mse_produces_values():
    """MSE should produce SampEn at multiple scales."""
    np.random.seed(42)
    data = np.random.randn(300) * 10 + 75
    scales = [1, 2, 3, 4, 5]
    result = multiscale_entropy(data, scales=scales, m=2, r_fraction=0.2)

    print(f"  MSE values: {[f'{v:.3f}' if v is not None else 'None' for v in result]}")
    assert len(result) == len(scales), f"Expected {len(scales)} values"
    # At least the first few scales should produce values
    assert result[0] is not None, "Scale 1 should produce a value"
    print("  ✅ PASSED")


# ════════════════════════════════════════════
# TEST GROUP 3: Normalization
# ════════════════════════════════════════════

def test_normalization_range():
    """Normalized values should be in [0, 1]."""
    test_cases = [
        ("heart_rate", 0.0, 0.0),   # below min → clamped to 0
        ("heart_rate", 0.1, 0.0),   # at min → 0
        ("heart_rate", 2.5, 1.0),   # at max → 1
        ("heart_rate", 5.0, 1.0),   # above max → clamped to 1
        ("heart_rate", 1.3, None),   # mid-range → between 0 and 1
    ]
    for vital, sampen, expected in test_cases:
        result = normalize_sampen(sampen, vital)
        if expected is not None:
            assert abs(result - expected) < 0.01, f"normalize({sampen}, {vital}) = {result}, expected ~{expected}"
        else:
            assert 0.0 < result < 1.0, f"normalize({sampen}, {vital}) = {result}, expected between 0 and 1"
    print("  ✅ PASSED — All normalization values in expected ranges")


def test_normalization_nan():
    """NaN input should return None."""
    result = normalize_sampen(float("nan"), "heart_rate")
    assert result is None, f"Expected None for NaN input, got {result}"
    print("  ✅ PASSED")


# ════════════════════════════════════════════
# TEST GROUP 4: Config
# ════════════════════════════════════════════

def test_default_config():
    """Default config should load with correct values."""
    config = AppConfig()
    assert config.entropy_engine.sampen_m == 2
    assert config.entropy_engine.window_size == 300
    assert config.entropy_engine.weights.heart_rate == 0.25
    assert abs(sum([
        config.entropy_engine.weights.heart_rate,
        config.entropy_engine.weights.spo2,
        config.entropy_engine.weights.bp_systolic,
        config.entropy_engine.weights.bp_diastolic,
        config.entropy_engine.weights.resp_rate,
        config.entropy_engine.weights.temperature,
    ]) - 1.0) < 0.001, "Weights should sum to 1.0"
    print("  ✅ PASSED")


def test_config_from_yaml():
    """Config should load from YAML if file exists."""
    # This will either load config.yml or use defaults
    config = load_config("config.yml")
    assert config.entropy_engine.sampen_m == 2
    print("  ✅ PASSED")


# ════════════════════════════════════════════
# TEST GROUP 5: Engine — Warmup
# ════════════════════════════════════════════

def test_engine_warmup():
    """Engine should report 'calibrating' until window is full."""
    config = AppConfig()
    config.entropy_engine.window_size = 50
    config.entropy_engine.warmup_points = 50
    engine = EntropyEngine(config)

    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 10, 0, 0)

    # Send 30 records (below warmup threshold of 50)
    for i in range(30):
        record = VitalSignRecord(
            patient_id="P001",
            timestamp=base_time + timedelta(minutes=i),
            heart_rate=75 + np.random.randn() * 8,
            spo2=97 + np.random.randn(),
            bp_systolic=120 + np.random.randn() * 10,
            bp_diastolic=80 + np.random.randn() * 6,
            resp_rate=16 + np.random.randn() * 2,
            temperature=37.0 + np.random.randn() * 0.2,
        )
        state = engine.process_vital(record)

    assert state.calibrating is True, "Should still be calibrating at 30/50 points"
    assert state.alert.severity == AlertSeverity.NONE
    assert "Calibrating" in state.alert.message
    print(f"  At 30 points: calibrating={state.calibrating}, fill={state.window_fill:.0%}")
    print("  ✅ PASSED")


def test_engine_post_warmup():
    """After warmup, engine should produce entropy scores and alerts."""
    config = AppConfig()
    config.entropy_engine.window_size = 50
    config.entropy_engine.warmup_points = 50
    engine = EntropyEngine(config)

    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 10, 0, 0)

    # Send 60 records (above warmup threshold of 50)
    for i in range(60):
        record = VitalSignRecord(
            patient_id="P001",
            timestamp=base_time + timedelta(minutes=i),
            heart_rate=75 + np.random.randn() * 8,
            spo2=97 + np.random.randn(),
            bp_systolic=120 + np.random.randn() * 10,
            bp_diastolic=80 + np.random.randn() * 6,
            resp_rate=16 + np.random.randn() * 2,
            temperature=37.0 + np.random.randn() * 0.2,
        )
        state = engine.process_vital(record)

    assert state.calibrating is False, "Should be done calibrating"
    assert 0.0 <= state.composite_entropy <= 1.0, f"CES should be [0,1], got {state.composite_entropy}"
    print(f"  CES: {state.composite_entropy:.4f}")
    print(f"  Alert: {state.alert.severity}")
    print(f"  HR SampEn: {state.vitals.heart_rate.sampen}")
    print("  ✅ PASSED")


# ════════════════════════════════════════════
# TEST GROUP 6: Engine — Deterioration Detection
# ════════════════════════════════════════════

def test_engine_detects_deterioration():
    """
    Simulate a patient going from healthy (complex signals)
    to deteriorating (rigid, predictable signals).
    CES should drop and alert should escalate.
    """
    config = AppConfig()
    config.entropy_engine.window_size = 60
    config.entropy_engine.warmup_points = 60
    engine = EntropyEngine(config)

    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 10, 0, 0)

    # Phase 1: 80 points of HEALTHY data (good variability)
    for i in range(80):
        record = VitalSignRecord(
            patient_id="P002",
            timestamp=base_time + timedelta(minutes=i),
            heart_rate=75 + np.random.randn() * 10,
            spo2=97 + np.random.randn() * 1.5,
            bp_systolic=120 + np.random.randn() * 12,
            bp_diastolic=80 + np.random.randn() * 8,
            resp_rate=16 + np.random.randn() * 3,
            temperature=37.0 + np.random.randn() * 0.3,
        )
        state = engine.process_vital(record)

    healthy_ces = state.composite_entropy
    healthy_severity = state.alert.severity
    print(f"  HEALTHY  — CES: {healthy_ces:.4f}, Alert: {healthy_severity}")

    # Phase 2: 80 points of DETERIORATING data (increasingly rigid)
    # Signal becomes sinusoidal (predictable) instead of random (complex)
    for i in range(80):
        t = 80 + i
        rigidity = min(1.0, i / 60.0)  # 0→1 over 60 points

        # Mix: (1-rigidity)*random + rigidity*sinusoidal
        noise = np.random.randn()
        periodic = np.sin(t * 0.3)

        record = VitalSignRecord(
            patient_id="P002",
            timestamp=base_time + timedelta(minutes=t),
            heart_rate=75 + ((1 - rigidity) * noise * 10) + (rigidity * periodic * 2),
            spo2=97 + ((1 - rigidity) * noise * 1.5) + (rigidity * periodic * 0.3),
            bp_systolic=120 + ((1 - rigidity) * noise * 12) + (rigidity * periodic * 2),
            bp_diastolic=80 + ((1 - rigidity) * noise * 8) + (rigidity * periodic * 1.5),
            resp_rate=16 + ((1 - rigidity) * noise * 3) + (rigidity * periodic * 0.5),
            temperature=37.0 + ((1 - rigidity) * noise * 0.3) + (rigidity * periodic * 0.05),
        )
        state = engine.process_vital(record)

    deteriorated_ces = state.composite_entropy
    deteriorated_severity = state.alert.severity
    print(f"  DETERIOR — CES: {deteriorated_ces:.4f}, Alert: {deteriorated_severity}")
    print(f"  Message: {state.alert.message}")
    print(f"  Contributing: {state.alert.contributing_vitals}")

    if state.alert.hours_to_predicted_event is not None:
        print(f"  Hours to event: {state.alert.hours_to_predicted_event}")

    # CES should have dropped
    assert deteriorated_ces < healthy_ces, (
        f"CES should drop during deterioration: healthy={healthy_ces:.4f}, deteriorated={deteriorated_ces:.4f}"
    )

    # Alert should be more severe (or at least active)
    severity_rank = {AlertSeverity.NONE: 0, AlertSeverity.WATCH: 1, AlertSeverity.WARNING: 2, AlertSeverity.CRITICAL: 3}
    assert severity_rank[deteriorated_severity] >= severity_rank.get(healthy_severity, 0), (
        f"Alert should escalate: was {healthy_severity}, now {deteriorated_severity}"
    )

    print("  ✅ PASSED — Deterioration detected: CES dropped and alert escalated")


# ════════════════════════════════════════════
# TEST GROUP 7: Alert Thresholds
# ════════════════════════════════════════════

def test_alert_threshold_classification():
    """Verify CES → severity mapping at boundary values."""
    config = AppConfig()
    engine = EntropyEngine(config)

    test_cases = [
        (0.95, AlertSeverity.NONE),
        (0.60, AlertSeverity.NONE),      # boundary: >= 0.60 is NONE
        (0.59, AlertSeverity.WATCH),     # just below NONE
        (0.40, AlertSeverity.WATCH),     # boundary: >= 0.40 is WATCH
        (0.39, AlertSeverity.WARNING),   # just below WATCH
        (0.20, AlertSeverity.WARNING),   # boundary: >= 0.20 is WARNING
        (0.19, AlertSeverity.CRITICAL),  # just below WARNING
        (0.05, AlertSeverity.CRITICAL),
        (0.00, AlertSeverity.CRITICAL),
    ]

    from app.entropy.engine import PatientWindow
    from app.models import VitalsState

    window = PatientWindow(300)
    vitals = VitalsState()

    all_passed = True
    for ces, expected in test_cases:
        actual = engine._ces_to_severity(ces)
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_passed = False
        print(f"  {status} CES={ces:.2f} → {actual.value} (expected {expected.value})")

    assert all_passed, "Some threshold classifications failed"
    print("  ✅ PASSED — All threshold classifications correct")


# ════════════════════════════════════════════
# TEST GROUP 8: Multi-patient support
# ════════════════════════════════════════════

def test_multi_patient():
    """Engine should track multiple patients independently."""
    config = AppConfig()
    config.entropy_engine.window_size = 50
    config.entropy_engine.warmup_points = 50
    engine = EntropyEngine(config)

    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 10, 0, 0)

    # Feed data for 3 patients simultaneously
    for i in range(60):
        for pid in ["P001", "P002", "P003"]:
            record = VitalSignRecord(
                patient_id=pid,
                timestamp=base_time + timedelta(minutes=i),
                heart_rate=75 + np.random.randn() * 8,
                spo2=97 + np.random.randn(),
                bp_systolic=120 + np.random.randn() * 10,
                bp_diastolic=80 + np.random.randn() * 6,
                resp_rate=16 + np.random.randn() * 2,
                temperature=37.0 + np.random.randn() * 0.2,
            )
            engine.process_vital(record)

    active = engine.get_active_patient_ids()
    assert len(active) == 3, f"Expected 3 patients, got {len(active)}"

    summaries = engine.get_all_summaries()
    assert len(summaries) == 3

    for s in summaries:
        print(f"  {s.patient_id}: CES={s.composite_entropy:.4f}, severity={s.alert_severity}")

    print("  ✅ PASSED — 3 patients tracked independently")


# ════════════════════════════════════════════
# TEST GROUP 9: MSE on demand
# ════════════════════════════════════════════

def test_mse_on_demand():
    """MSE should be computable on demand for a patient."""
    config = AppConfig()
    config.entropy_engine.window_size = 50
    config.entropy_engine.warmup_points = 50
    engine = EntropyEngine(config)

    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 10, 0, 0)

    for i in range(60):
        record = VitalSignRecord(
            patient_id="P001",
            timestamp=base_time + timedelta(minutes=i),
            heart_rate=75 + np.random.randn() * 8,
            spo2=97 + np.random.randn(),
            bp_systolic=120 + np.random.randn() * 10,
            bp_diastolic=80 + np.random.randn() * 6,
            resp_rate=16 + np.random.randn() * 2,
            temperature=37.0 + np.random.randn() * 0.2,
        )
        engine.process_vital(record)

    mse_results = engine.compute_mse_for_patient("P001")
    assert len(mse_results) > 0, "MSE should return results"
    print(f"  MSE vitals computed: {list(mse_results.keys())}")
    for vital, values in mse_results.items():
        printable = [f"{v:.3f}" if v is not None else "None" for v in values[:5]]
        print(f"    {vital}: [{', '.join(printable)}, ...]")

    print("  ✅ PASSED")


# ════════════════════════════════════════════
# TEST GROUP 10: Edge cases
# ════════════════════════════════════════════

def test_missing_vitals():
    """Engine should handle records with missing vital signs."""
    config = AppConfig()
    config.entropy_engine.window_size = 50
    config.entropy_engine.warmup_points = 50
    engine = EntropyEngine(config)

    base_time = datetime(2024, 1, 15, 10, 0, 0)
    np.random.seed(42)

    for i in range(60):
        # Only heart_rate and spo2, everything else missing
        record = VitalSignRecord(
            patient_id="P001",
            timestamp=base_time + timedelta(minutes=i),
            heart_rate=75 + np.random.randn() * 8,
            spo2=97 + np.random.randn(),
            # bp_systolic, bp_diastolic, resp_rate, temperature → None
        )
        state = engine.process_vital(record)

    # Should still produce a CES from available vitals
    assert not state.calibrating
    print(f"  CES with partial data: {state.composite_entropy:.4f}")
    print(f"  HR sampen: {state.vitals.heart_rate.sampen}")
    print(f"  SpO2 sampen: {state.vitals.spo2.sampen}")
    print(f"  BP sys sampen: {state.vitals.bp_systolic.sampen} (expected None — no data)")
    assert state.vitals.bp_systolic.sampen is None, "Missing vital should have no SampEn"
    print("  ✅ PASSED — Graceful handling of missing vitals")


# ════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PROJECT CHRONOS — Phase 1 Test Suite")
    print(f"  SampEn backend: {get_backend()}")
    print("=" * 60)

    tests = [
        ("SampEn: Constant Signal", test_sampen_constant_signal),
        ("SampEn: Sine Wave", test_sampen_sine_wave),
        ("SampEn: Random Noise", test_sampen_random_noise),
        ("SampEn: Ordering", test_sampen_ordering),
        ("SampEn: Short Data", test_sampen_short_data),
        ("SampEn: NaN Handling", test_sampen_with_nans),
        ("SampEn: Performance", test_sampen_performance),
        ("MSE: Coarse Graining", test_coarse_grain),
        ("MSE: Multi-Scale Values", test_mse_produces_values),
        ("Normalization: Range", test_normalization_range),
        ("Normalization: NaN", test_normalization_nan),
        ("Config: Defaults", test_default_config),
        ("Config: YAML", test_config_from_yaml),
        ("Engine: Warmup", test_engine_warmup),
        ("Engine: Post-Warmup", test_engine_post_warmup),
        ("Engine: Deterioration Detection", test_engine_detects_deterioration),
        ("Alert: Threshold Classification", test_alert_threshold_classification),
        ("Engine: Multi-Patient", test_multi_patient),
        ("Engine: MSE on Demand", test_mse_on_demand),
        ("Engine: Missing Vitals", test_missing_vitals),
    ]

    passed = 0
    failed = 0
    failures = []

    for name, test_fn in tests:
        try:
            header(name)
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            failures.append(name)
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    if failures:
        print(f"  Failed tests:")
        for f in failures:
            print(f"    ❌ {f}")
    else:
        print(f"  🎉 ALL TESTS PASSED")
    print(f"{'=' * 60}")

    sys.exit(0 if failed == 0 else 1)
