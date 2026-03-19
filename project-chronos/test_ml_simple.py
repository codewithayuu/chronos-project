"""Simple test script for Person 2 ML Runtime components."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def test_predictor():
    print("\n--- TEST: DeteriorationPredictor ---")
    from app.ml.predictor import DeteriorationPredictor

    pred = DeteriorationPredictor("data/models")
    features = np.random.randn(45)
    result = pred.predict(features)

    print(f"  Available: {pred.available}")
    print(f"  Result: {result}")

    if not pred.available:
        print("  OK - Graceful degradation, models not found")
        assert result is None
    else:
        print(f"  OK - Models loaded, risk_4h: {result['risk_4h']:.3f}")
    print("  PASSED")


def test_classifier():
    print("\n--- TEST: SyndromeClassifier ---")
    from app.ml.classifier import SyndromeClassifier

    clf = SyndromeClassifier("data/models")
    features = np.random.randn(45)
    result = clf.predict(features)

    print(f"  Available: {clf.available}")
    print(f"  Result: {result}")

    if not clf.available:
        print("  OK - Graceful degradation")
        assert result is None
    print("  PASSED")


def test_fusion_no_ml():
    print("\n--- TEST: DecisionFusion - No ML ---")
    from app.core.fusion import DecisionFusion

    fusion = DecisionFusion()
    result = fusion.fuse(
        ces_adjusted=0.35, ces_slope_6h=-0.002,
        ml_risk_1h=None, ml_risk_4h=None, ml_risk_8h=None,
        drug_masking=True, news2_score=3,
    )

    print(f"  FRS: {result['final_risk_score']}")
    print(f"  Severity: {result['final_severity']}")
    print(f"  ML Available: {result['ml_available']}")
    print(f"  Component Risks: {result['component_risks']}")

    assert result["ml_available"] is False
    assert result["component_risks"]["ml"] == 0.0
    assert 0 <= result["final_risk_score"] <= 100
    assert result["final_severity"] in ("NONE", "WATCH", "WARNING", "CRITICAL")
    print("  PASSED")


def test_fusion_with_ml():
    print("\n--- TEST: DecisionFusion - With ML ---")
    from app.core.fusion import DecisionFusion

    fusion = DecisionFusion()
    result = fusion.fuse(
        ces_adjusted=0.35, ces_slope_6h=-0.002,
        ml_risk_1h=0.15, ml_risk_4h=0.60, ml_risk_8h=0.82,
        drug_masking=True, news2_score=3,
    )

    print(f"  FRS: {result['final_risk_score']}")
    print(f"  Severity: {result['final_severity']}")
    print(f"  ML Available: {result['ml_available']}")
    print(f"  Time Estimate: {result['time_to_event_estimate']}")

    assert result["ml_available"] is True
    assert result["component_risks"]["ml"] == 0.6
    print("  PASSED")


def test_fusion_overrides():
    print("\n--- TEST: DecisionFusion - Override Scenarios ---")
    from app.core.fusion import DecisionFusion

    fusion = DecisionFusion()

    # All safe
    r = fusion.fuse(ces_adjusted=0.75, ces_slope_6h=0.0, ml_risk_1h=0.05,
                    ml_risk_4h=0.10, ml_risk_8h=0.15, drug_masking=False, news2_score=1)
    print(f"  All safe:       FRS={r['final_risk_score']:3d}, Sev={r['final_severity']}")
    assert r["final_severity"] == "NONE"

    # Entropy critical
    r = fusion.fuse(ces_adjusted=0.15, ces_slope_6h=-0.003, ml_risk_1h=0.05,
                    ml_risk_4h=0.10, ml_risk_8h=0.15, drug_masking=False, news2_score=1)
    print(f"  Entropy crit:   FRS={r['final_risk_score']:3d}, Sev={r['final_severity']}, Override={r['override_applied']}")
    assert r["final_severity"] == "CRITICAL"

    # ML high
    r = fusion.fuse(ces_adjusted=0.70, ces_slope_6h=0.0, ml_risk_1h=0.40,
                    ml_risk_4h=0.90, ml_risk_8h=0.95, drug_masking=False, news2_score=1)
    print(f"  ML high:        FRS={r['final_risk_score']:3d}, Sev={r['final_severity']}, Override={r['override_applied']}")
    assert r["final_severity"] in ("WARNING", "CRITICAL")

    # Drug masked
    r = fusion.fuse(ces_adjusted=0.35, ces_slope_6h=-0.001, ml_risk_1h=0.20,
                    ml_risk_4h=0.55, ml_risk_8h=0.65, drug_masking=True, news2_score=2)
    print(f"  Drug masked:    FRS={r['final_risk_score']:3d}, Sev={r['final_severity']}, Override={r['override_applied']}")
    assert r["final_severity"] in ("WARNING", "CRITICAL")

    # No ML available
    r = fusion.fuse(ces_adjusted=0.45, ces_slope_6h=-0.001, ml_risk_1h=None,
                    ml_risk_4h=None, ml_risk_8h=None, drug_masking=False, news2_score=3)
    print(f"  No ML:          FRS={r['final_risk_score']:3d}, Sev={r['final_severity']}")
    assert r["ml_available"] is False

    print("  PASSED")


def test_detectors():
    print("\n--- TEST: DetectorBank ---")
    from app.core.detectors import DetectorBank

    detectors = DetectorBank()
    results = detectors.run_all(
        entropy_state={
            "ces_adjusted": 0.35, "ces_slope_6h": -0.002, "warmup_complete": True,
            "sampen_hr": 0.3, "sampen_bp_sys": 0.4, "sampen_rr": 0.2, "sampen_spo2": 0.5,
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
    assert len(results) == 8

    active = [d for d in results if d["active"]]
    print(f"  Active: {[d['detector_name'] for d in active]}")

    for d in results:
        assert "detector_name" in d
        assert "active" in d
        assert "severity" in d

    et = next(d for d in results if d["detector_name"] == "entropy_threshold")
    assert et["active"] is True, "Entropy threshold should fire at CES 0.35"
    print(f"  Entropy: {et['severity']} - {et['message']}")

    sd = next(d for d in results if d["detector_name"] == "silent_decline")
    assert sd["active"] is True, "Silent decline should fire"
    print(f"  Silent:  {sd['severity']} - {sd['message']}")

    dm = next(d for d in results if d["detector_name"] == "drug_masking")
    assert dm["active"] is True, "Drug masking should fire"
    print(f"  Drug:    {dm['severity']} - {dm['message']}")

    print("  PASSED")


def test_detectors_normal():
    print("\n--- TEST: DetectorBank - All Normal ---")
    from app.core.detectors import DetectorBank

    detectors = DetectorBank()
    results = detectors.run_all(
        entropy_state={
            "ces_adjusted": 0.80, "ces_slope_6h": 0.001, "warmup_complete": True,
            "sampen_hr": 1.2, "sampen_bp_sys": 1.1, "sampen_rr": 1.0, "sampen_spo2": 0.9,
        },
        drug_state={"drug_masking": False, "active_drugs": []},
        vitals={"hr": 72, "bp_sys": 120, "bp_dia": 78, "rr": 14, "spo2": 98, "temp": 36.8},
        ml_predictions={"deterioration": {"risk_4h": 0.08}, "syndrome": {"primary_syndrome": "Stable"}},
        fusion={"final_severity": "NONE", "final_risk_score": 12},
    )

    active = [d for d in results if d["active"]]
    non_recovery = [d for d in active if d["detector_name"] != "recovery"]
    print(f"  Active (non-recovery): {[d['detector_name'] for d in non_recovery]}")
    assert len(non_recovery) == 0
    print("  PASSED")


def test_detectors_warmup():
    print("\n--- TEST: DetectorBank - Warmup ---")
    from app.core.detectors import DetectorBank

    detectors = DetectorBank()
    results = detectors.run_all(
        entropy_state={
            "ces_adjusted": 0.35, "ces_slope_6h": -0.003, "warmup_complete": False,
            "sampen_hr": 0.3, "sampen_bp_sys": 0.4, "sampen_rr": 0.2, "sampen_spo2": 0.5,
        },
        drug_state={"drug_masking": False, "active_drugs": []},
        vitals={"hr": 84, "bp_sys": 112, "bp_dia": 72, "rr": 17, "spo2": 96, "temp": 37.2},
        ml_predictions={"deterioration": None, "syndrome": None},
        fusion={"final_severity": "NONE", "final_risk_score": 0},
    )

    dq = next(d for d in results if d["detector_name"] == "data_quality")
    assert dq["active"] is True, "Data quality should flag warmup"
    print(f"  Data quality: {dq['message']}")

    et = next(d for d in results if d["detector_name"] == "entropy_threshold")
    assert et["active"] is False, "Entropy threshold should not fire during warmup"
    print("  PASSED")


def test_models_import():
    print("\n--- TEST: Pydantic Models ---")
    from app.models import (
        MLPredictions, FusionResult, DetectorResult, Recommendations,
        DeteriorationRisk, SyndromeClassification, DisagreementInfo,
        SuggestedTest, DriverDetail,
    )

    ml = MLPredictions()
    assert ml.warmup_mode is True
    print(f"  MLPredictions: warmup_mode={ml.warmup_mode}")

    fusion = FusionResult()
    assert fusion.final_severity == "NONE"
    print(f"  FusionResult: severity={fusion.final_severity}")

    det = DetectorResult(detector_name="test")
    assert det.active is False
    print(f"  DetectorResult: name={det.detector_name}")

    rec = Recommendations()
    assert len(rec.interventions) == 0
    print(f"  Recommendations: ok")

    risk = DeteriorationRisk(
        risk_1h=0.12, risk_4h=0.58, risk_8h=0.81,
        model_confidence="high",
        top_drivers=[DriverDetail(feature="ces_slope_6h", description="Declining entropy", importance=0.31)]
    )
    assert risk.risk_4h == 0.58
    print(f"  DeteriorationRisk: risk_4h={risk.risk_4h}")
    print("  PASSED")


def test_contracts():
    print("\n--- TEST: Interface Contracts ---")
    from app.core.fusion import DecisionFusion
    from app.core.detectors import DetectorBank

    fusion = DecisionFusion()
    result = fusion.fuse(
        ces_adjusted=0.38, ces_slope_6h=-0.002,
        ml_risk_1h=0.12, ml_risk_4h=0.58, ml_risk_8h=0.81,
        drug_masking=True, news2_score=3
    )

    required = ["final_risk_score", "final_severity", "time_to_event_estimate",
                "component_risks", "ml_available", "override_applied", "disagreement"]
    for key in required:
        assert key in result, f"Missing: {key}"
    print(f"  Fusion contract: OK ({len(required)} keys)")

    assert isinstance(result["final_risk_score"], int)
    for comp in ("entropy", "trend", "ml", "masking", "news2"):
        assert comp in result["component_risks"]

    bank = DetectorBank()
    results = bank.run_all(
        entropy_state={"ces_adjusted": 0.5, "ces_slope_6h": 0.0, "warmup_complete": True,
                       "sampen_hr": 1.0, "sampen_bp_sys": 1.0, "sampen_rr": 1.0, "sampen_spo2": 1.0},
        drug_state={"drug_masking": False, "active_drugs": []},
        vitals={"hr": 80, "bp_sys": 120, "bp_dia": 75, "rr": 16, "spo2": 97, "temp": 37.0},
        ml_predictions={"deterioration": None, "syndrome": None},
        fusion={"final_severity": "NONE"}
    )

    assert len(results) == 8
    det_keys = ["detector_name", "active", "severity", "message", "contributing_factors", "recommended_action"]
    for d in results:
        for key in det_keys:
            assert key in d, f"Missing: {key}"
    print(f"  Detector contract: OK (8 detectors, {len(det_keys)} keys)")
    print("  PASSED")


if __name__ == "__main__":
    print("PROJECT CHRONOS - ML Runtime Test Suite (Person 2)")
    print("=" * 55)

    tests = [
        test_predictor,
        test_classifier,
        test_fusion_no_ml,
        test_fusion_with_ml,
        test_fusion_overrides,
        test_detectors,
        test_detectors_normal,
        test_detectors_warmup,
        test_models_import,
        test_contracts,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED - {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 55)
    print(f"RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 55)

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed! ML Runtime is ready for integration.")
