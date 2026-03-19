"""
Phase 3 Tests — Data Generator, Pipeline, and REST API.

Run with:  python -m tests.test_phase3   (from project-chronos directory)
    or:    cd project-chronos && python tests/test_phase3.py
"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
from app.config import AppConfig
from app.models import VitalSignRecord, DrugEffect, AlertSeverity
from app.data.generator import DataGenerator, PatientCase
from app.data.replay import ReplayService
from app.pipeline import ChronosPipeline

def header(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

# ════════════════════════════════════


# TEST GROUP 1: Data Generator


# ════════════════════════════════════

def test_hero_case_1_generation():
    """Hero Case 1 (Silent Sepsis) should generate valid data."""
    rng = np.random.RandomState(42)
    base_time = datetime(2024, 1, 15, 8, 0, 0)
    case = DataGenerator.hero_case_1(base_time, rng)

    assert case.patient_id == "BED-01"
    assert len(case.records) == 720, f"Expected 720 records, got {len(case.records)}"
    assert len(case.drug_events) >= 1, "Should have at least 1 drug event"

    # Check first and last records
    first = case.records[0]
    last = case.records[-1]
    assert first.patient_id == "BED-01"
    assert first.timestamp == base_time
    assert last.timestamp == base_time + timedelta(minutes=719)

    # Check vital ranges are physiological
    for r in case.records:
        assert 30 <= r.heart_rate <= 200, f"HR out of range: {r.heart_rate}"
        assert 50 <= r.spo2 <= 100, f"SpO2 out of range: {r.spo2}"
        assert 40 <= r.bp_systolic <= 250, f"BP out of range: {r.bp_systolic}"

    # Check that late-stage vitals show deterioration
    late_hrs = [r.heart_rate for r in case.records[660:720]]
    early_hrs = [r.heart_rate for r in case.records[100:160]]
    assert np.mean(late_hrs) > np.mean(early_hrs), "Late HR should be higher than early HR"

    print(f"  Case: {case.name}")
    print(f"  Records: {len(case.records)}")
    print(f"  Drug events: {len(case.drug_events)}")
    print(f"  Early HR mean: {np.mean(early_hrs):.1f}, Late HR mean: {np.mean(late_hrs):.1f}")
    print("  ✅ PASSED")

def test_demo_dataset_generation():
    """Full demo dataset should generate hero + filler patients."""
    cases = DataGenerator.generate_demo_dataset(
        seed=42, num_filler=5, duration_minutes=720
    )

    assert len(cases) == 8, f"Expected 8 cases (3 hero + 5 filler), got {len(cases)}"

    # Check all have unique patient IDs
    ids = [c.patient_id for c in cases]
    assert len(set(ids)) == len(ids), f"Duplicate patient IDs found: {ids}"

    # Hero cases
    assert cases[0].patient_id == "BED-01"
    assert cases[1].patient_id == "BED-02"
    assert cases[2].patient_id == "BED-03"

    # All cases should have 720 records
    for case in cases:
        assert len(case.records) == 720, f"{case.patient_id}: expected 720 records"

    print(f"  Generated {len(cases)} cases: {ids}")
    print("  ✅ PASSED")

def test_data_determinism():
    """Same seed should produce identical datasets."""
    d1 = DataGenerator.generate_demo_dataset(seed=42, num_filler=2)
    d2 = DataGenerator.generate_demo_dataset(seed=42, num_filler=2)

    for c1, c2 in zip(d1, d2):
        for r1, r2 in zip(c1.records[:10], c2.records[:10]):
            assert r1.heart_rate == r2.heart_rate, "Non-deterministic generation"
    print("  ✅ PASSED — Deterministic generation confirmed")

# ══════════════════════════════════════


# TEST GROUP 2: Pipeline


# ══════════════════════════════════════

def _make_test_pipeline(window_size=60):
    """Create a pipeline with small window for fast testing."""
    config = AppConfig()
    config.entropy_engine.window_size = window_size
    config.entropy_engine.warmup_points = window_size
    return ChronosPipeline(config), config

def test_pipeline_process_vital():
    """Pipeline should process vitals through all layers."""
    pipeline, _ = _make_test_pipeline(50)
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
        state = pipeline.process_vital(record)

    assert not state.calibrating, "Should be done calibrating"
    assert 0 <= state.composite_entropy <= 1
    print(f"  CES: {state.composite_entropy:.4f}")
    print(f"  Alert: {state.alert.severity}")
    print("  ✅ PASSED")

def test_pipeline_drug_integration():
    """Pipeline should integrate drugs into processing."""
    pipeline, _ = _make_test_pipeline(50)
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    np.random.seed(42)

    # Feed 60 points
    for i in range(60):
        pipeline.process_vital(VitalSignRecord(
            patient_id="P001",
            timestamp=base_time + timedelta(minutes=i),
            heart_rate=75 + np.random.randn() * 8,
            spo2=97 + np.random.randn(),
            bp_systolic=120 + np.random.randn() * 10,
            bp_diastolic=80 + np.random.randn() * 6,
            resp_rate=16 + np.random.randn() * 2,
            temperature=37.0 + np.random.randn() * 0.2,
        ))

    # Add a drug
    pipeline.add_drug("P001", DrugEffect(
        drug_name="Propofol",
        drug_class="sedative",
        dose=50.0,
        unit="mcg/kg/min",
        start_time=base_time + timedelta(minutes=55),
    ))

    drugs = pipeline.get_patient_drugs("P001")
    assert len(drugs) == 1, f"Expected 1 drug, got {len(drugs)}"
    assert drugs[0].drug_name == "Propofol"

    # Process one more vital — should include drug in processing
    state = pipeline.process_vital(VitalSignRecord(
        patient_id="P001",
        timestamp=base_time + timedelta(minutes=61),
        heart_rate=72 + np.random.randn() * 5,
        spo2=96,
        bp_systolic=115,
        bp_diastolic=76,
        resp_rate=14,
        temperature=37.0,
    ))

    assert len(state.active_drugs) == 1, "State should include active drug"
    print(f"  Drug: {state.active_drugs[0].drug_name}")
    print(f"  CES raw: {state.composite_entropy_raw:.4f}, adjusted: {state.composite_entropy:.4f}")
    print("  ✅ PASSED")

def test_pipeline_history():
    """Pipeline should store state history."""
    pipeline, _ = _make_test_pipeline(30)
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    np.random.seed(42)

    for i in range(40):
        pipeline.process_vital(VitalSignRecord(
            patient_id="P001",
            timestamp=base_time + timedelta(minutes=i),
            heart_rate=75 + np.random.randn() * 8,
            spo2=97 + np.random.randn(),
            bp_systolic=120 + np.random.randn() * 10,
            bp_diastolic=80 + np.random.randn() * 6,
            resp_rate=16 + np.random.randn() * 2,
            temperature=37.0,
        ))

    history = pipeline.get_patient_history("P001", hours=1)
    assert len(history) == 40, f"Expected 40 history entries, got {len(history)}"

    # Check history is ordered by time
    for i in range(1, len(history)):
        assert history[i].timestamp >= history[i - 1].timestamp

    print(f"  History entries: {len(history)}")
    print("  ✅ PASSED")

def test_pipeline_multi_patient():
    """Pipeline should track multiple patients independently."""
    pipeline, _ = _make_test_pipeline(30)
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    np.random.seed(42)

    for i in range(35):
        for pid in ["P001", "P002", "P003"]:
            pipeline.process_vital(VitalSignRecord(
                patient_id=pid,
                timestamp=base_time + timedelta(minutes=i),
                heart_rate=75 + np.random.randn() * 8,
                spo2=97 + np.random.randn(),
                bp_systolic=120 + np.random.randn() * 10,
                bp_diastolic=80 + np.random.randn() * 6,
                resp_rate=16 + np.random.randn() * 2,
                temperature=37.0,
            ))

    ids = pipeline.get_all_patient_ids()
    assert len(ids) == 3
    summaries = pipeline.get_all_summaries()
    assert len(summaries) == 3

    for s in summaries:
        print(f"  {s.patient_id}: CES={s.composite_entropy:.4f}")
    print("  ✅ PASSED")

def test_pipeline_system_health():
    """System health should return valid info."""
    pipeline, _ = _make_test_pipeline(30)
    health = pipeline.get_system_health()

    assert health["status"] == "ok"
    assert health["evidence_engine_ready"] is True
    assert health["drug_database_size"] > 0
    print(f"  Health: {health}")
    print("  ✅ PASSED")

# ══════════════════════════════════════


# TEST GROUP 3: Replay Service


# ══════════════════════════════════════

def test_replay_service_tick():
    """Replay service should feed data to pipeline via tick()."""
    pipeline, config = _make_test_pipeline(50)
    replay = ReplayService(pipeline, config)

    # Generate small dataset
    cases = DataGenerator.generate_demo_dataset(seed=42, num_filler=1, duration_minutes=100)
    replay.load_cases(cases)

    # Tick 60 times (past warmup)
    for _ in range(60):
        replay.tick()

    # Should have 4 patients (3 hero + 1 filler) tracked
    ids = pipeline.get_all_patient_ids()
    assert len(ids) == 4, f"Expected 4 patients, got {len(ids)}: {ids}"

    # Each patient should have state
    for pid in ids:
        state = pipeline.get_patient_state(pid)
        assert state is not None, f"No state for {pid}"

    # After 60 ticks with window_size=50, patients should be past warmup
    bed01 = pipeline.get_patient_state("BED-01")
    assert not bed01.calibrating, "BED-01 should be past calibration at 60 points"

    print(f"  Patients after 60 ticks: {ids}")
    print(f"  BED-01 CES: {bed01.composite_entropy:.4f}")
    print(f"  Replay progress: {replay.progress:.1%}")
    print("  ✅ PASSED")

def test_replay_hero_case_deterioration():
    """Hero case should show clear deterioration when replayed through pipeline."""
    config = AppConfig()
    config.entropy_engine.window_size = 60
    config.entropy_engine.warmup_points = 60
    pipeline = ChronosPipeline(config)
    replay = ReplayService(pipeline, config)

    # Only load hero case 1
    rng = np.random.RandomState(42)
    base_time = datetime(2024, 1, 15, 8, 0, 0)
    hero1 = DataGenerator.hero_case_1(base_time, rng)
    replay.load_cases([hero1])

    # Tick through healthy phase (first 120 minutes, past 60-point warmup)
    for _ in range(120):
        replay.tick()

    healthy_state = pipeline.get_patient_state("BED-01")
    healthy_ces = healthy_state.composite_entropy
    print(f"  At minute 120 (healthy): CES={healthy_ces:.4f}, severity={healthy_state.alert.severity}")

    # Tick through deterioration (to minute 600)
    for _ in range(480):
        replay.tick()

    deteriorated_state = pipeline.get_patient_state("BED-01")
    deteriorated_ces = deteriorated_state.composite_entropy
    print(f"  At minute 600 (crisis): CES={deteriorated_ces:.4f}, severity={deteriorated_state.alert.severity}")

    # CES should have dropped
    assert deteriorated_ces < healthy_ces, (
        f"CES should drop: healthy={healthy_ces:.4f} → deteriorated={deteriorated_ces:.4f}"
    )

    # Alert should be active with elevated severity
    assert deteriorated_state.alert.active, "Alert should be active during crisis"
    severity_rank = {"NONE": 0, "WATCH": 1, "WARNING": 2, "CRITICAL": 3}
    assert severity_rank[deteriorated_state.alert.severity.value] >= 1, (
        f"Alert severity should be at least WATCH, got {deteriorated_state.alert.severity}"
    )

    # Should have interventions
    if deteriorated_state.interventions:
        print(f"  Top intervention: {deteriorated_state.interventions[0].action[:60]}...")

    print("  ✅ PASSED — Hero case shows clear deterioration")

# ════════════════════════════════════════


# TEST GROUP 4: REST API


# ════════════════════════════════════════

def test_api_endpoints():
    """Test all REST API endpoints using FastAPI TestClient."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.routes import router

    # Create test app with pipeline
    config = AppConfig()
    config.entropy_engine.window_size = 30
    config.entropy_engine.warmup_points = 30
    pipeline = ChronosPipeline(config)

    app = FastAPI()
    app.state.pipeline = pipeline
    app.include_router(router)

    client = TestClient(app)

    # ── 1. Health check ──
    resp = client.get("/api/v1/system/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    print(f"  GET /system/health → {data['status']}")

    # ── 2. No patients yet ──
    resp = client.get("/api/v1/patients")
    assert resp.status_code == 200
    assert resp.json() == []
    print(f"  GET /patients → [] (empty)")

    # ── 3. POST vitals ──
    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    for i in range(35):
        vital = {
            "patient_id": "P001",
            "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
            "heart_rate": round(75 + np.random.randn() * 8, 1),
            "spo2": round(97 + np.random.randn(), 1),
            "bp_systolic": round(120 + np.random.randn() * 10, 1),
            "bp_diastolic": round(80 + np.random.randn() * 6, 1),
            "resp_rate": round(16 + np.random.randn() * 2, 1),
            "temperature": 37.0,
        }
        resp = client.post("/api/v1/vitals", json=vital)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["patient_id"] == "P001"

    print(f"  POST /vitals × 35 → accepted")

    # ── 4. GET patients (should have 1 now) ──
    resp = client.get("/api/v1/patients")
    assert resp.status_code == 200
    patients = resp.json()
    assert len(patients) == 1
    assert patients[0]["patient_id"] == "P001"
    print(f"  GET /patients → [{patients[0]['patient_id']}]")

    # ── 5. GET single patient ──
    resp = client.get("/api/v1/patients/P001")
    assert resp.status_code == 200
    pdata = resp.json()
    assert pdata["patient_id"] == "P001"
    assert "composite_entropy" in pdata
    assert "vitals" in pdata
    print(f"  GET /patients/P001 → CES={pdata['composite_entropy']:.4f}")

    # ── 6. GET unknown patient → 404 ──
    resp = client.get("/api/v1/patients/UNKNOWN")
    assert resp.status_code == 404
    print(f"  GET /patients/UNKNOWN → 404 ✓")

    # ── 7. GET history ──
    resp = client.get("/api/v1/patients/P001/history?hours=1")
    assert resp.status_code == 200
    history = resp.json()
    assert len(history) == 35
    print(f"  GET /patients/P001/history → {len(history)} snapshots")

    # ── 8. POST drug ──
    drug_data = {
        "drug_name": "Propofol",
        "drug_class": "sedative",
        "dose": 50.0,
        "unit": "mcg/kg/min",
    }
    resp = client.post("/api/v1/patients/P001/drugs", json=drug_data)
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"
    print(f"  POST /patients/P001/drugs → recorded")

    # ── 9. GET drugs ──
    resp = client.get("/api/v1/patients/P001/drugs")
    assert resp.status_code == 200
    drugs = resp.json()
    assert len(drugs) == 1
    assert drugs[0]["drug_name"] == "Propofol"
    print(f"  GET /patients/P001/drugs → [{drugs[0]['drug_name']}]")

    # ── 10. GET alerts ──
    resp = client.get("/api/v1/alerts")
    assert resp.status_code == 200
    # May or may not have alerts depending on CES
    print(f"  GET /alerts → {len(resp.json())} alerts")

    print("  ✅ PASSED — All API endpoints working")

def test_api_alert_acknowledge():
    """Test alert acknowledgment workflow."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.routes import router

    config = AppConfig()
    config.entropy_engine.window_size = 30
    config.entropy_engine.warmup_points = 30
    pipeline = ChronosPipeline(config)

    app = FastAPI()
    app.state.pipeline = pipeline
    app.include_router(router)

    client = TestClient(app)

    # Feed deteriorating data to trigger an alert
    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 10, 0, 0)

    # Healthy phase
    for i in range(35):
        client.post("/api/v1/vitals", json={
            "patient_id": "P001",
            "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
            "heart_rate": round(75 + np.random.randn() * 8, 1),
            "spo2": round(97 + np.random.randn(), 1),
            "bp_systolic": round(120 + np.random.randn() * 10, 1),
            "bp_diastolic": round(80 + np.random.randn() * 6, 1),
            "resp_rate": round(16 + np.random.randn() * 2, 1),
            "temperature": 37.0,
        })

    # Deteriorating phase (rigid signals)
    for i in range(35):
        t = 35 + i
        periodic = np.sin(t * 0.3)
        client.post("/api/v1/vitals", json={
            "patient_id": "P001",
            "timestamp": (base_time + timedelta(minutes=t)).isoformat(),
            "heart_rate": round(75 + periodic * 1.5, 1),
            "spo2": round(97 + periodic * 0.2, 1),
            "bp_systolic": round(120 + periodic * 2, 1),
            "bp_diastolic": round(80 + periodic * 1, 1),
            "resp_rate": round(16 + periodic * 0.3, 1),
            "temperature": 37.0,
        })

    # Check alerts
    resp = client.get("/api/v1/alerts")
    alerts = resp.json()
    print(f"  Active alerts: {len(alerts)}")

    if len(alerts) > 0:
        alert_id = alerts[0]["alert_id"]
        # Acknowledge
        resp = client.post(
            f"/api/v1/alerts/{alert_id}/acknowledge",
            json={"acknowledged_by": "Dr. Meera"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "acknowledged"
        print(f"  Acknowledged alert {alert_id}")

        # Verify acknowledged
        resp = client.get("/api/v1/alerts")
        updated = resp.json()
        acked = [a for a in updated if a["alert_id"] == alert_id]
        if acked:
            assert acked[0]["acknowledged"] is True
            print(f"  Alert {alert_id} acknowledged=True ✓")

    # Try acknowledging nonexistent alert
    resp = client.post(
        "/api/v1/alerts/FAKE-9999/acknowledge",
        json={"acknowledged_by": "test"},
    )
    assert resp.status_code == 404
    print(f"  Acknowledge FAKE-9999 → 404 ✓")

    print("  ✅ PASSED")

# ════════════════════════════════════════


# RUNNER


# ══════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PROJECT CHRONOS — Phase 3 Test Suite")
    print("  Data Generator + Pipeline + REST API")
    print("=" * 60)

    tests = [
        ("Data: Hero Case 1", test_hero_case_1_generation),
        ("Data: Demo Dataset", test_demo_dataset_generation),
        ("Data: Determinism", test_data_determinism),
        ("Pipeline: Process Vital", test_pipeline_process_vital),
        ("Pipeline: Drug Integration", test_pipeline_drug_integration),
        ("Pipeline: History", test_pipeline_history),
        ("Pipeline: Multi-Patient", test_pipeline_multi_patient),
        ("Pipeline: System Health", test_pipeline_system_health),
        ("Replay: Tick Service", test_replay_service_tick),
        ("Replay: Hero Deterioration", test_replay_hero_case_deterioration),
        ("API: All Endpoints", test_api_endpoints),
        ("API: Alert Acknowledgment", test_api_alert_acknowledge),
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
