"""
Test script for Person 2's ML Runtime components.

Exercises all classes with synthetic data to verify:
  - Graceful degradation when models are missing
  - Fusion math with all override scenarios
  - Detector logic for all 8 detectors
  - Correct interface shapes for Person 3 integration

Run: python test_ml_runtime.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def test_predictor():
    """Test DeteriorationPredictor with graceful degradation."""
    print("\n" + "=" * 60)
    print("TEST: DeteriorationPredictor")
    print("=" * 60)

    from app.ml.predictor import DeteriorationPredictor

    # Test with non-existent models (should degrade gracefully)
    pred = DeteriorationPredictor("data/models")
    features = np.random.randn(45)
    result = pred.predict(features)

    print(f"  Available: {pred.available}")
    print(f"  Result: {result}")

    if not pred.available:
        print("  ✓ Graceful degradation — models not found, returned None")
        assert result is None, "Should return None when models unavailable"
    else:
        print(f"  ✓ Models loaded — risk_4h: {result['risk_4h']:.3f}")
        assert "risk_1h" in result
        assert "risk_4h" in result
        assert "risk_8h" in result
        assert "model_confidence" in result
        assert "top_drivers" in result
        assert isinstance(result["top_drivers"], list)

    print("  PASSED ✓")


def test_classifier():
    """Test SyndromeClassifier with graceful degradation."""
    print("\n" + "=" * 60)
    print("TEST: SyndromeClassifier")
    print("=" * 60)

    from app.ml.classifier import SyndromeClassifier

    clf = SyndromeClassifier("data/models")
    features = np.random.randn(45)
    result = clf.predict(features)

    print(f"  Available: {clf.available}")
    print(f"  Result: {result}")

    if not clf.available:
        print("  ✓ Graceful degradation — model not found, returned None")
        assert result is None, "Should return None when model unavailable"
    else:
        print(f"  ✓ Model loaded — primary: {result['primary_syndrome']}")
        assert "primary_syndrome" in result
        assert "primary_confidence" in result
        assert "all_probabilities" in result
        assert "disclaimer" in result

    print("  PASSED ✓")


def test_fusion_no_ml():
    """Test DecisionFusion in entropy-only mode (ML unavailable)."""
    print("\n" + "=" * 60)
    print("TEST: DecisionFusion — No ML")
    print("=" * 60)

    from app.core.fusion import DecisionFusion

    fusion = DecisionFusion()
    result = fusion.fuse(
        ces_adjusted=0.35,
        ces_slope_6h=-0.002,
        ml_risk_1h=None,
        ml_risk_4h=None,
        ml_risk_8h=None,
        drug_masking=True,
        news2_score=3,
    )

    print(f"  FRS: {result['final_risk_score']}")
    print(f"  Severity: {result['final_severity']}")
    print(f"  ML Available: {result['ml_available']}")
    print(f"  Component Risks: {result['component_risks']}")
    print(f"  Override: {result['override_applied']}")

    assert result["ml_available"] is False, "ML should not be available"
    assert result["component_risks"]["ml"] == 0.0, "ML risk should be 0 when unavailable"
    assert result["final_risk_score"] >= 0 and result["final_risk_score"] <= 100
    assert result["final_severity"] in ("NONE", "WATCH", "WARNING", "CRITICAL")

    print("  PASSED ✓")


def test_fusion_with_ml():
    """Test DecisionFusion with ML predictions."""
    print("\n" + "=" * 60)
    print("TEST: DecisionFusion — With ML")
    print("=" * 60)

    from app.core.fusion import DecisionFusion

    fusion = DecisionFusion()
    result = fusion.fuse(
        ces_adjusted=0.35,
        ces_slope_6h=-0.002,
        ml_risk_1h=0.15,
        ml_risk_4h=0.60,
        ml_risk_8h=0.82,
        drug_masking=True,
        news2_score=3,
    )

    print(f"  FRS: {result['final_risk_score']}")
    print(f"  Severity: {result['final_severity']}")
    print(f"  ML Available: {result['ml_available']}")
    print(f"  Time Estimate: {result['time_to_event_estimate']}")
    print(f"  Component Risks: {result['component_risks']}")

    assert result["ml_available"] is True
    assert result["component_risks"]["ml"] == 0.6
    assert result["final_risk_score"] > 0

    print("  PASSED ✓")


def test_fusion_override_scenarios():
    """Test all critical fusion override scenarios."""
    print("\n" + "=" * 60)
    print("TEST: DecisionFusion — Override Scenarios")
    print("=" * 60)

    from app.core.fusion import DecisionFusion

    fusion = DecisionFusion()

    # Scenario 1: All safe
    r = fusion.fuse(ces_adjusted=0.75, ces_slope_6h=0.0, ml_risk_1h=0.05,
                    ml_risk_4h=0.10, ml_risk_8h=0.15, drug_masking=False, news2_score=1)
    print(f"  All safe:            FRS={r['final_risk_score']:3d}, Severity={r['final_severity']}")
    assert r["final_severity"] == "NONE", f"Expected NONE, got {r['final_severity']}"

    # Scenario 2: Entropy critical
    r = fusion.fuse(ces_adjusted=0.15, ces_slope_6h=-0.003, ml_risk_1h=0.05,
                    ml_risk_4h=0.10, ml_risk_8h=0.15, drug_masking=False, news2_score=1)
    print(f"  Entropy critical:    FRS={r['final_risk_score']:3d}, Severity={r['final_severity']}, Override={r['override_applied']}")
    assert r["final_severity"] == "CRITICAL", f"Expected CRITICAL, got {r['final_severity']}"

    # Scenario 3: ML high, entropy fine
    r = fusion.fuse(ces_adjusted=0.70, ces_slope_6h=0.0, ml_risk_1h=0.40,
                    ml_risk_4h=0.90, ml_risk_8h=0.95, drug_masking=False, news2_score=1)
    print(f"  ML high:             FRS={r['final_risk_score']:3d}, Severity={r['final_severity']}, Override={r['override_applied']}")
    assert r["final_severity"] in ("WARNING", "CRITICAL"), f"Expected WARNING/CRITICAL, got {r['final_severity']}"

    # Scenario 4: Drug masked + low CES
    r = fusion.fuse(ces_adjusted=0.35, ces_slope_6h=-0.001, ml_risk_1h=0.20,
                    ml_risk_4h=0.55, ml_risk_8h=0.65, drug_masking=True, news2_score=2)
    print(f"  Drug masked+low CES: FRS={r['final_risk_score']:3d}, Severity={r['final_severity']}, Override={r['override_applied']}")
    assert r["final_severity"] in ("WARNING", "CRITICAL"), f"Expected WARNING+, got {r['final_severity']}"

    # Scenario 5: Disagreement — entropy high, ML low
    r = fusion.fuse(ces_adjusted=0.25, ces_slope_6h=-0.003, ml_risk_1h=0.05,
                    ml_risk_4h=0.10, ml_risk_8h=0.15, drug_masking=False, news2_score=2)
    print(f"  Disagreement:        FRS={r['final_risk_score']:3d}, Severity={r['final_severity']}, Disagree={r['disagreement'] is not None}")

    # Scenario 6: No ML available
    r = fusion.fuse(ces_adjusted=0.45, ces_slope_6h=-0.001, ml_risk_1h=None,
                    ml_risk_4h=None, ml_risk_8h=None, drug_masking=False, news2_score=3)
    print(f"  No ML:               FRS={r['final_risk_score']:3d}, Severity={r['final_severity']}")
    assert r["ml_available"] is False

    print("  PASSED ✓")


def test_detectors():
    """Test DetectorBank with a realistic patient state."""
    print("\n" + "=" * 60)
    print("TEST: DetectorBank")
    print("=" * 60)

    from app.core.detectors import DetectorBank

    detectors = DetectorBank()

    results = detectors.run_all(
        entropy_state={
            "ces_adjusted": 0.35,
            "ces_slope_6h": -0.002,
            "warmup_complete": True,
            "sampen_hr": 0.3,
            "sampen_bp_sys": 0.4,
            "sampen_rr": 0.2,
            "sampen_spo2": 0.5,
        },
        drug_state={
            "drug_masking": True,
            "active_drugs": [{"drug_name": "Norepinephrine", "dose": 0.08, "unit": "mcg/kg/min"}],
        },
        vitals={"hr": 84, "bp_sys": 112, "bp_dia": 72, "rr": 17, "spo2": 96, "temp": 37.2},
        ml_predictions={
            "deterioration": {"risk_4h": 0.6},
            "syndrome": {"primary_syndrome": "Hemodynamic Instability"},
        },
        fusion={"final_severity": "WARNING", "final_risk_score": 52},
    )

    print(f"  Total detectors: {len(results)}")
    assert len(results) == 8, f"Expected 8 detectors, got {len(results)}"

    active = [d for d in results if d["active"]]
    print(f"  Active detectors: {[d['detector_name'] for d in active]}")

    # Verify each result has required keys
    for d in results:
        assert "detector_name" in d
        assert "active" in d
        assert "severity" in d
        assert "message" in d
        assert "contributing_factors" in d
        assert "recommended_action" in d

    # Entropy threshold should fire (CES = 0.35 < 0.40)
    entropy_det = next(d for d in results if d["detector_name"] == "entropy_threshold")
    assert entropy_det["active"] is True, "Entropy threshold should be active at CES 0.35"
    print(f"  Entropy threshold: {entropy_det['severity']} — {entropy_det['message']}")

    # Silent decline should fire (slope negative, vitals normal)
    silent_det = next(d for d in results if d["detector_name"] == "silent_decline")
    assert silent_det["active"] is True, "Silent decline should fire"
    print(f"  Silent decline: {silent_det['severity']} — {silent_det['message']}")

    # Drug masking should fire
    drug_det = next(d for d in results if d["detector_name"] == "drug_masking")
    assert drug_det["active"] is True, "Drug masking should fire"
    print(f"  Drug masking: {drug_det['severity']} — {drug_det['message']}")

    print("  PASSED ✓")


def test_detectors_all_normal():
    """Test DetectorBank when everything is normal."""
    print("\n" + "=" * 60)
    print("TEST: DetectorBank — All Normal")
    print("=" * 60)

    from app.core.detectors import DetectorBank

    detectors = DetectorBank()
    results = detectors.run_all(
        entropy_state={
            "ces_adjusted": 0.80,
            "ces_slope_6h": 0.001,
            "warmup_complete": True,
            "sampen_hr": 1.2,
            "sampen_bp_sys": 1.1,
            "sampen_rr": 1.0,
            "sampen_spo2": 0.9,
        },
        drug_state={"drug_masking": False, "active_drugs": []},
        vitals={"hr": 72, "bp_sys": 120, "bp_dia": 78, "rr": 14, "spo2": 98, "temp": 36.8},
        ml_predictions={"deterioration": {"risk_4h": 0.08}, "syndrome": {"primary_syndrome": "Stable"}},
        fusion={"final_severity": "NONE", "final_risk_score": 12},
    )

    active = [d for d in results if d["active"]]
    print(f"  Active detectors: {[d['detector_name'] for d in active]}")
    # Should have few or no active detectors
    # Recovery might fire if conditions are right
    non_recovery = [d for d in active if d["detector_name"] not in ("recovery",)]
    assert len(non_recovery) == 0, f"No non-recovery detectors should fire, got {[d['detector_name'] for d in non_recovery]}"

    print("  PASSED ✓")


def test_detectors_warmup():
    """Test DetectorBank during warmup (most should be inactive)."""
    print("\n" + "=" * 60)
    print("TEST: DetectorBank — Warmup Mode")
    print("=" * 60)

    from app.core.detectors import DetectorBank

    detectors = DetectorBank()
    results = detectors.run_all(
        entropy_state={
            "ces_adjusted": 0.35,
            "ces_slope_6h": -0.003,
            "warmup_complete": False,
            "sampen_hr": 0.3,
            "sampen_bp_sys": 0.4,
            "sampen_rr": 0.2,
            "sampen_spo2": 0.5,
        },
        drug_state={"drug_masking": False, "active_drugs": []},
        vitals={"hr": 84, "bp_sys": 112, "bp_dia": 72, "rr": 17, "spo2": 96, "temp": 37.2},
        ml_predictions={"deterioration": None, "syndrome": None},
        fusion={"final_severity": "NONE", "final_risk_score": 0},
    )

    active = [d for d in results if d["active"]]
    print(f"  Active during warmup: {[d['detector_name'] for d in active]}")

    # Data quality should fire (warmup_complete = False)
    dq = next(d for d in results if d["detector_name"] == "data_quality")
    assert dq["active"] is True, "Data quality should flag warmup"
    print(f"  Data quality: {dq['message']}")

    # Entropy threshold should NOT fire during warmup
    et = next(d for d in results if d["detector_name"] == "entropy_threshold")
    assert et["active"] is False, "Entropy threshold should not fire during warmup"

    print("  PASSED ✓")


def test_models_import():
    """Test that new Pydantic models are importable and work."""
    print("\n" + "=" * 60)
    print("TEST: Pydantic Models")
    print("=" * 60)

    from app.models import (
        MLPredictions,
        FusionResult,
        DetectorResult,
        Recommendations,
        DeteriorationRisk,
        SyndromeClassification,
        DisagreementInfo,
        SuggestedTest,
        DriverDetail,
    )

    # Test default construction
    ml = MLPredictions()
    print(f"  MLPredictions: warmup_mode={ml.warmup_mode}")
    assert ml.warmup_mode is True

    fusion = FusionResult()
    print(f"  FusionResult: severity={fusion.final_severity}")
    assert fusion.final_severity == "NONE"

    det = DetectorResult(detector_name="test")
    print(f"  DetectorResult: name={det.detector_name}, active={det.active}")
    assert det.active is False

    rec = Recommendations()
    print(f"  Recommendations: interventions={len(rec.interventions)}")
    assert len(rec.interventions) == 0

    # Test populated construction
    risk = DeteriorationRisk(
        risk_1h=0.12, risk_4h=0.58, risk_8h=0.81,
        model_confidence="high",
        top_drivers=[DriverDetail(feature="ces_slope_6h", description="Declining entropy", importance=0.31)]
    )
    print(f"  DeteriorationRisk: risk_4h={risk.risk_4h}")
    assert risk.risk_4h == 0.58

    print("  PASSED ✓")


def test_interface_contracts():
    """
    Verify that all output shapes match the contracts defined for Person 3.
    """
    print("\n" + "=" * 60)
    print("TEST: Interface Contracts")
    print("=" * 60)

    from app.core.fusion import DecisionFusion
    from app.core.detectors import DetectorBank

    # Fusion contract
    fusion = DecisionFusion()
    result = fusion.fuse(
        ces_adjusted=0.38, ces_slope_6h=-0.002,
        ml_risk_1h=0.12, ml_risk_4h=0.58, ml_risk_8h=0.81,
        drug_masking=True, news2_score=3
    )

    required_keys = [
        "final_risk_score", "final_severity", "time_to_event_estimate",
        "component_risks", "ml_available", "override_applied", "disagreement"
    ]
    for key in required_keys:
        assert key in result, f"Missing key in fusion result: {key}"
    print(f"  Fusion contract: ✓ ({len(required_keys)} required keys present)")

    assert isinstance(result["final_risk_score"], int)
    assert result["final_severity"] in ("NONE", "WATCH", "WARNING", "CRITICAL")
    assert isinstance(result["component_risks"], dict)
    for comp in ("entropy", "trend", "ml", "masking", "news2"):
        assert comp in result["component_risks"], f"Missing component: {comp}"

    # Detector contract
    bank = DetectorBank()
    results = bank.run_all(
        entropy_state={"ces_adjusted": 0.5, "ces_slope_6h": 0.0, "warmup_complete": True,
                       "sampen_hr": 1.0, "sampen_bp_sys": 1.0, "sampen_rr": 1.0, "sampen_spo2": 1.0},
        drug_state={"drug_masking": False, "active_drugs": []},
        vitals={"hr": 80, "bp_sys": 120, "bp_dia": 75, "rr": 16, "spo2": 97, "temp": 37.0},
        ml_predictions={"deterioration": None, "syndrome": None},
        fusion={"final_severity": "NONE"}
    )

    assert len(results) == 8, f"Expected 8 detectors, got {len(results)}"
    det_keys = ["detector_name", "active", "severity", "message", "contributing_factors", "recommended_action"]
    for d in results:
        for key in det_keys:
            assert key in d, f"Missing key in detector result: {key}"
    print(f"  Detector contract: ✓ ({len(results)} detectors, {len(det_keys)} keys each)")

    print("  PASSED ✓")


if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║  PROJECT CHRONOS — ML Runtime Test Suite (Person 2)     ║")
    print("╚" + "═" * 58 + "╝")

    tests = [
        test_predictor,
        test_classifier,
        test_fusion_no_ml,
        test_fusion_with_ml,
        test_fusion_override_scenarios,
        test_detectors,
        test_detectors_all_normal,
        test_detectors_warmup,
        test_models_import,
        test_interface_contracts,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED ✗ — {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 All tests passed! ML Runtime is ready for integration.")
