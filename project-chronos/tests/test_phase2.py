"""
Phase 2 Comprehensive Tests — Drug Awareness Filter + Evidence Engine

Run with:  python -m tests.test_phase2   (from project-chronos directory)
    or:    cd project-chronos && python tests/test_phase2.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
from app.config import AppConfig, load_config
from app.models import (
    VitalSignRecord,
    PatientState,
    DrugEffect,
    AlertSeverity,
    TrendDirection,
    VitalDetail,
    VitalsState,
    AlertState,
    Intervention,
)
from app.drugs.database import DrugDatabase
from app.drugs.filter import DrugFilter
from app.evidence.cases import (
    generate_synthetic_cases,
    extract_feature_matrix,
    FEATURE_NAMES,
    NUM_FEATURES,
    HistoricalCase,
)
from app.evidence.engine import EvidenceEngine
from app.entropy.engine import EntropyEngine


def header(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ════════════════════════════════════════════
# TEST GROUP 1: Drug Database
# ════════════════════════════════════════════

def test_drug_database_loads():
    """Drug database should load all 15 drugs from JSON."""
    db = DrugDatabase("data/drug_database.json")
    assert len(db) == 15, f"Expected 15 drugs, got {len(db)}"
    print(f"  Loaded {len(db)} drugs")
    print(f"  Classes: {db.get_all_classes()}")
    print("  ✅ PASSED")


def test_drug_lookup_by_name():
    """Should look up drugs case-insensitively."""
    db = DrugDatabase("data/drug_database.json")

    prop = db.lookup("Propofol")
    assert prop is not None, "Propofol should exist"
    assert prop.drug_class == "sedative"
    assert prop.expected_hr_effect == "decrease"
    print(f"  Propofol: class={prop.drug_class}, HR effect={prop.expected_hr_effect} ({prop.expected_hr_magnitude})")

    nore = db.lookup("norepinephrine")
    assert nore is not None, "Norepinephrine should exist (case-insensitive)"
    assert nore.drug_class == "vasopressor"
    print(f"  Norepinephrine: class={nore.drug_class}, BP effect={nore.expected_bp_effect} ({nore.expected_bp_magnitude})")

    missing = db.lookup("Tylenol")
    assert missing is None, "Unknown drug should return None"
    print(f"  Tylenol: {missing} (correctly None)")

    print("  ✅ PASSED")


def test_drug_affected_vitals():
    """Should correctly identify which vitals a drug affects."""
    db = DrugDatabase("data/drug_database.json")

    prop_vitals = db.get_affected_vitals("Propofol")
    assert "heart_rate" in prop_vitals, "Propofol should affect HR"
    assert "resp_rate" in prop_vitals, "Propofol should affect RR"
    assert "bp_systolic" in prop_vitals, "Propofol should affect BP"
    print(f"  Propofol affects: {prop_vitals}")

    met_vitals = db.get_affected_vitals("Metoprolol")
    assert "heart_rate" in met_vitals, "Metoprolol should affect HR"
    assert "resp_rate" not in met_vitals, "Metoprolol should NOT affect RR"
    print(f"  Metoprolol affects: {met_vitals}")

    print("  ✅ PASSED")


def test_drug_expected_change():
    """Should return correct expected magnitude of change."""
    db = DrugDatabase("data/drug_database.json")

    hr_change = db.get_expected_change("Propofol", "heart_rate")
    assert hr_change == -10.0, f"Propofol HR change should be -10.0, got {hr_change}"

    bp_change = db.get_expected_change("Norepinephrine", "bp_systolic")
    assert bp_change == 20.0, f"Norepi BP change should be 20.0, got {bp_change}"

    temp_change = db.get_expected_change("Propofol", "temperature")
    assert temp_change is None, "No drug should affect temperature"

    print(f"  Propofol→HR: {hr_change}, Norepi→BP: {bp_change}, Propofol→Temp: {temp_change}")
    print("  ✅ PASSED")


# ════════════════════════════════════════════
# TEST GROUP 2: Drug Filter — Suppression
# ════════════════════════════════════════════

def _make_patient_state(
    ces: float,
    hr_val: float, hr_sampen_norm: float,
    bp_val: float, bp_sampen_norm: float,
    rr_val: float, rr_sampen_norm: float,
    spo2_val: float, spo2_sampen_norm: float,
    drugs: list = None,
    hr_trend=TrendDirection.STABLE,
) -> PatientState:
    """Helper to construct a PatientState for testing."""
    return PatientState(
        patient_id="TEST001",
        timestamp=datetime(2024, 1, 15, 14, 0, 0),
        vitals=VitalsState(
            heart_rate=VitalDetail(value=hr_val, sampen=0.5, sampen_normalized=hr_sampen_norm, trend=hr_trend),
            spo2=VitalDetail(value=spo2_val, sampen=0.5, sampen_normalized=spo2_sampen_norm),
            bp_systolic=VitalDetail(value=bp_val, sampen=0.5, sampen_normalized=bp_sampen_norm),
            bp_diastolic=VitalDetail(value=bp_val * 0.67, sampen=0.5, sampen_normalized=bp_sampen_norm),
            resp_rate=VitalDetail(value=rr_val, sampen=0.5, sampen_normalized=rr_sampen_norm),
            temperature=VitalDetail(value=37.0, sampen=0.5, sampen_normalized=0.7),
        ),
        composite_entropy=ces,
        composite_entropy_raw=ces,
        active_drugs=drugs or [],
        alert=AlertState(
            active=(ces < 0.60),
            severity=AlertSeverity.WATCH if ces < 0.60 else AlertSeverity.NONE,
            message="Test alert",
        ),
        calibrating=False,
    )


def test_filter_no_drugs():
    """With no drugs, filter should not change anything."""
    db = DrugDatabase("data/drug_database.json")
    filt = DrugFilter(db)

    state = _make_patient_state(
        ces=0.55,
        hr_val=80, hr_sampen_norm=0.6,
        bp_val=120, bp_sampen_norm=0.6,
        rr_val=16, rr_sampen_norm=0.6,
        spo2_val=97, spo2_sampen_norm=0.7,
    )
    baselines = {"heart_rate": 80, "bp_systolic": 120, "bp_diastolic": 80, "resp_rate": 16, "spo2": 97}

    result = filt.apply(state, baselines)
    assert abs(result.composite_entropy - 0.55) < 0.01, "CES should not change without drugs"
    assert not result.alert.drug_masked
    print(f"  No drugs: CES {state.composite_entropy_raw:.2f} → {result.composite_entropy:.2f} (unchanged)")
    print("  ✅ PASSED")


def test_filter_drug_suppression():
    """
    Patient on Metoprolol with expected HR drop.
    HR dropped from 85 (baseline) to 70 (current) — consistent with Metoprolol.
    Filter should reduce HR's CES contribution → adjusted CES should be HIGHER.
    """
    db = DrugDatabase("data/drug_database.json")
    filt = DrugFilter(db)

    drug_start = datetime(2024, 1, 15, 13, 0, 0)  # 1 hour ago
    drugs = [DrugEffect(
        drug_name="Metoprolol",
        drug_class="beta_blocker",
        dose=5.0,
        unit="mg",
        start_time=drug_start,
    )]

    state = _make_patient_state(
        ces=0.42,
        hr_val=70, hr_sampen_norm=0.35,      # HR dropped and low entropy
        bp_val=110, bp_sampen_norm=0.55,      # BP slightly low
        rr_val=16, rr_sampen_norm=0.65,
        spo2_val=97, spo2_sampen_norm=0.7,
        drugs=drugs,
    )
    baselines = {"heart_rate": 85, "bp_systolic": 120, "bp_diastolic": 80, "resp_rate": 16, "spo2": 97}

    result = filt.apply(state, baselines)

    print(f"  Metoprolol: CES {state.composite_entropy_raw:.4f} → {result.composite_entropy:.4f}")
    print(f"  Drug masked: {result.alert.drug_masked}")
    print(f"  Severity: {result.alert.severity}")
    print(f"  Message: {result.alert.message[:100]}...")

    assert result.composite_entropy >= state.composite_entropy_raw, (
        f"Adjusted CES should be >= raw when drug explains the change. "
        f"Raw={state.composite_entropy_raw:.4f}, Adjusted={result.composite_entropy:.4f}"
    )
    assert not result.alert.drug_masked, "Should NOT be flagged as masked (change is explained)"
    print("  ✅ PASSED — Drug-explained change suppressed, CES adjusted upward")


def test_filter_drug_masking_detected():
    """
    Patient on Propofol. Entropy is dropping (trend=FALLING, low sampen)
    but vital VALUES appear stable (drug is propping them up).
    Filter should flag as drug_masked.
    """
    db = DrugDatabase("data/drug_database.json")
    filt = DrugFilter(db)

    drug_start = datetime(2024, 1, 15, 13, 55, 0)  # 5 min ago
    drugs = [DrugEffect(
        drug_name="Propofol",
        drug_class="sedative",
        dose=50.0,
        unit="mcg/kg/min",
        start_time=drug_start,
    )]

    state = _make_patient_state(
        ces=0.30,
        hr_val=78, hr_sampen_norm=0.25,       # Low entropy
        bp_val=118, bp_sampen_norm=0.30,       # Low entropy but value stable
        rr_val=15, rr_sampen_norm=0.20,        # Low entropy but value stable
        spo2_val=96, spo2_sampen_norm=0.35,
        drugs=drugs,
        hr_trend=TrendDirection.FALLING,        # Entropy is falling
    )
    baselines = {"heart_rate": 80, "bp_systolic": 120, "bp_diastolic": 80, "resp_rate": 16, "spo2": 97}

    result = filt.apply(state, baselines)

    print(f"  Propofol masking: CES raw={result.composite_entropy_raw:.4f}, adjusted={result.composite_entropy:.4f}")
    print(f"  Drug masked: {result.alert.drug_masked}")
    print(f"  Severity: {result.alert.severity}")
    print(f"  Message: {result.alert.message[:120]}...")

    assert result.alert.drug_masked is True, "Should detect drug masking"
    print("  ✅ PASSED — Drug masking correctly detected")


def test_filter_drug_outside_window():
    """Drug given long ago (outside duration window) should not affect filtering."""
    db = DrugDatabase("data/drug_database.json")
    filt = DrugFilter(db)

    # Esmolol given 2 hours ago, but Esmolol duration is only 20 min
    drug_start = datetime(2024, 1, 15, 12, 0, 0)  # 2 hours ago
    drugs = [DrugEffect(
        drug_name="Esmolol",
        drug_class="beta_blocker",
        dose=500.0,
        unit="mcg/kg/min",
        start_time=drug_start,
    )]

    state = _make_patient_state(
        ces=0.45,
        hr_val=65, hr_sampen_norm=0.40,
        bp_val=100, bp_sampen_norm=0.50,
        rr_val=16, rr_sampen_norm=0.60,
        spo2_val=96, spo2_sampen_norm=0.65,
        drugs=drugs,
    )
    baselines = {"heart_rate": 85, "bp_systolic": 120, "bp_diastolic": 80, "resp_rate": 16, "spo2": 97}

    result = filt.apply(state, baselines)

    print(f"  Expired Esmolol: CES {state.composite_entropy_raw:.4f} → {result.composite_entropy:.4f}")
    assert abs(result.composite_entropy - state.composite_entropy_raw) < 0.01, (
        "Expired drug should not affect CES"
    )
    print("  ✅ PASSED — Expired drug correctly ignored")


# ════════════════════════════════════════════
# TEST GROUP 3: Synthetic Case Generation
# ════════════════════════════════════════════

def test_case_generation():
    """Should generate the requested number of cases."""
    cases = generate_synthetic_cases(num_cases=500, seed=42)
    assert len(cases) > 0, "Should generate cases"
    print(f"  Generated {len(cases)} cases")

    # Check feature vector dimensions
    for case in cases[:5]:
        assert case.features.shape == (NUM_FEATURES,), f"Wrong feature shape: {case.features.shape}"

    # Check cluster distribution
    types = {}
    for c in cases:
        types[c.deterioration_type] = types.get(c.deterioration_type, 0) + 1
    print(f"  Distribution: {types}")

    print("  ✅ PASSED")


def test_case_feature_matrix():
    """Should produce a valid N×D feature matrix."""
    cases = generate_synthetic_cases(num_cases=100, seed=42)
    matrix = extract_feature_matrix(cases)
    assert matrix.shape == (len(cases), NUM_FEATURES)
    assert not np.any(np.isnan(matrix)), "No NaN values in feature matrix"
    print(f"  Matrix shape: {matrix.shape}")
    print(f"  Feature means: {np.mean(matrix, axis=0)[:5].round(2)}... (first 5)")
    print("  ✅ PASSED")


def test_case_determinism():
    """Same seed should produce identical cases."""
    cases1 = generate_synthetic_cases(num_cases=50, seed=42)
    cases2 = generate_synthetic_cases(num_cases=50, seed=42)
    m1 = extract_feature_matrix(cases1)
    m2 = extract_feature_matrix(cases2)
    assert np.allclose(m1, m2), "Same seed should produce identical features"
    print("  ✅ PASSED — Deterministic generation confirmed")


# ════════════════════════════════════════════
# TEST GROUP 4: Evidence Engine
# ════════════════════════════════════════════

def test_evidence_engine_builds():
    """Engine should build KD-Tree from synthetic cases."""
    engine = EvidenceEngine()
    engine.build()
    assert engine.is_ready, "Engine should be ready after build"
    assert engine.case_count == 500, f"Expected 500 cases, got {engine.case_count}"
    print(f"  Built with {engine.case_count} cases")
    print("  ✅ PASSED")


def test_evidence_engine_query_septic():
    """Query with sepsis-like features should return relevant interventions."""
    engine = EvidenceEngine()
    engine.build()

    # Create a sepsis-like patient state
    state = _make_patient_state(
        ces=0.25,
        hr_val=115, hr_sampen_norm=0.25,
        bp_val=75, bp_sampen_norm=0.20,
        rr_val=28, rr_sampen_norm=0.22,
        spo2_val=91, spo2_sampen_norm=0.30,
    )
    state.alert.severity = AlertSeverity.WARNING

    interventions = engine.query(state)

    print(f"  Sepsis-like patient → {len(interventions)} interventions:")
    for intv in interventions:
        print(f"    #{intv.rank}: {intv.action[:60]}... "
              f"(success={intv.historical_success_rate:.0%}, n={intv.similar_cases_count})")

    assert len(interventions) > 0, "Should return at least 1 intervention"
    assert interventions[0].historical_success_rate > 0, "First intervention should have success rate"
    assert interventions[0].similar_cases_count >= 5, "Should have minimum case count"
    print("  ✅ PASSED — Relevant interventions returned")


def test_evidence_engine_no_alert():
    """Engine should return empty list when patient has no alert."""
    engine = EvidenceEngine()
    engine.build()

    state = _make_patient_state(
        ces=0.80,
        hr_val=75, hr_sampen_norm=0.80,
        bp_val=120, bp_sampen_norm=0.75,
        rr_val=14, rr_sampen_norm=0.78,
        spo2_val=98, spo2_sampen_norm=0.85,
    )
    state.alert.severity = AlertSeverity.NONE

    interventions = engine.query(state)
    assert len(interventions) == 0, "No interventions for stable patient"
    print("  ✅ PASSED — No interventions for stable patient")


# ════════════════════════════════════════════
# TEST GROUP 5: End-to-End Pipeline
# ════════════════════════════════════════════

def test_full_pipeline_entropy_to_evidence():
    """
    Full pipeline: entropy engine → drug filter → evidence engine.
    Simulate a deteriorating patient on Propofol.
    """
    config = AppConfig()
    config.entropy_engine.window_size = 60
    config.entropy_engine.warmup_points = 60

    # Build all components
    entropy_engine = EntropyEngine(config)
    drug_db = DrugDatabase("data/drug_database.json")
    drug_filter = DrugFilter(drug_db, config)
    evidence_engine = EvidenceEngine(config)
    evidence_engine.build()

    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 10, 0, 0)

    # Phase 1: 70 healthy data points
    for i in range(70):
        record = VitalSignRecord(
            patient_id="P001",
            timestamp=base_time + timedelta(minutes=i),
            heart_rate=78 + np.random.randn() * 8,
            spo2=97 + np.random.randn(),
            bp_systolic=120 + np.random.randn() * 10,
            bp_diastolic=80 + np.random.randn() * 6,
            resp_rate=15 + np.random.randn() * 2,
            temperature=37.0 + np.random.randn() * 0.2,
        )
        state = entropy_engine.process_vital(record)

    healthy_ces = state.composite_entropy
    print(f"  HEALTHY phase: CES={healthy_ces:.4f}, severity={state.alert.severity}")

    # Phase 2: 80 deteriorating points (signals become rigid)
    # Also add Propofol administration at minute 100
    drug_start = base_time + timedelta(minutes=100)
    propofol = DrugEffect(
        drug_name="Propofol",
        drug_class="sedative",
        dose=50.0,
        unit="mcg/kg/min",
        start_time=drug_start,
    )

    for i in range(80):
        t = 70 + i
        rigidity = min(1.0, i / 50.0)
        noise = np.random.randn()
        periodic = np.sin(t * 0.3)

        record = VitalSignRecord(
            patient_id="P001",
            timestamp=base_time + timedelta(minutes=t),
            heart_rate=78 + ((1 - rigidity) * noise * 8) + (rigidity * periodic * 1.5),
            spo2=97 + ((1 - rigidity) * noise * 1) + (rigidity * periodic * 0.2),
            bp_systolic=120 + ((1 - rigidity) * noise * 10) + (rigidity * periodic * 2),
            bp_diastolic=80 + ((1 - rigidity) * noise * 6) + (rigidity * periodic * 1),
            resp_rate=15 + ((1 - rigidity) * noise * 2) + (rigidity * periodic * 0.5),
            temperature=37.0 + ((1 - rigidity) * noise * 0.2) + (rigidity * periodic * 0.05),
        )
        state = entropy_engine.process_vital(record)

    # Get baselines for drug filter
    baselines = entropy_engine.get_vital_baselines("P001")

    # Apply drug filter (patient on Propofol)
    state.active_drugs = [propofol]
    filtered_state = drug_filter.apply(state, baselines)

    print(f"  DETERIORATED: CES_raw={filtered_state.composite_entropy_raw:.4f}, "
          f"CES_adj={filtered_state.composite_entropy:.4f}")
    print(f"  Drug masked: {filtered_state.alert.drug_masked}")
    print(f"  Alert severity: {filtered_state.alert.severity}")

    # Apply evidence engine
    interventions = evidence_engine.query(filtered_state)
    filtered_state.interventions = interventions

    if interventions:
        print(f"  Interventions ({len(interventions)}):")
        for intv in interventions:
            print(f"    #{intv.rank}: {intv.action[:65]}... "
                  f"(success={intv.historical_success_rate:.0%})")
    else:
        print("  No interventions returned (alert may not be active)")

    # Verify the pipeline produced meaningful results
    assert filtered_state.composite_entropy < healthy_ces, "CES should have dropped"
    assert filtered_state.alert.active or filtered_state.alert.severity != AlertSeverity.NONE, (
        "Alert should be active for deteriorating patient"
    )

    print("\n  ✅ PASSED — Full pipeline: entropy → drug filter → evidence engine")


# ════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PROJECT CHRONOS — Phase 2 Test Suite")
    print("  Drug Awareness Filter + Evidence Engine")
    print("=" * 60)

    tests = [
        ("Drug DB: Loading", test_drug_database_loads),
        ("Drug DB: Lookup by Name", test_drug_lookup_by_name),
        ("Drug DB: Affected Vitals", test_drug_affected_vitals),
        ("Drug DB: Expected Change", test_drug_expected_change),
        ("Drug Filter: No Drugs", test_filter_no_drugs),
        ("Drug Filter: Suppression", test_filter_drug_suppression),
        ("Drug Filter: Masking Detection", test_filter_drug_masking_detected),
        ("Drug Filter: Expired Drug", test_filter_drug_outside_window),
        ("Cases: Generation", test_case_generation),
        ("Cases: Feature Matrix", test_case_feature_matrix),
        ("Cases: Determinism", test_case_determinism),
        ("Evidence: Build", test_evidence_engine_builds),
        ("Evidence: Sepsis Query", test_evidence_engine_query_septic),
        ("Evidence: No Alert", test_evidence_engine_no_alert),
        ("Pipeline: Full End-to-End", test_full_pipeline_entropy_to_evidence),
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
