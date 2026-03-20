
import os
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ..config import load_config
from ..core.manager import PatientManager
from ..data.generator import DataGenerator
from ..data.replay import ReplayService
from .websocket import ConnectionManager
from .routes import create_router

logger = logging.getLogger(__name__)


# ------------------------------------------
# Background Replay Task
# ------------------------------------------

async def replay_loop(app: FastAPI):
    """
    Background task: replays demo data through the pipeline
    and broadcasts updates via WebSocket.
    """
    manager: PatientManager = app.state.manager
    ws: ConnectionManager = app.state.ws_manager

    speed = int(os.environ.get("CHRONOS_SPEED", "60"))
    loop_replay = os.environ.get("CHRONOS_LOOP", "true").lower() == "true"

    print(f"[Replay] Preparing dataset from MIMIC-IV resources...")
    from ..data.synthetic_generator import MIMICIngester, generate_trajectory_simulations, select_hero_cases
    from ..data.generator import PatientCase, DrugEvent
    from ..models import VitalSignRecord
    from datetime import datetime, timedelta
    import numpy as np
    
    rng = np.random.RandomState(42)
    ingester = MIMICIngester("data/mimic-iv-demo")
    real_trajs = ingester.ingest()
    
    sim_trajs = generate_trajectory_simulations(patients_per_template=10, rng=rng)
    heroes = select_hero_cases(sim_trajs)
    
    all_trajs = []
    
    # 1. Add our 3 Hero Cases (Sepsis, Resp, Cardiac)
    for hero_name, traj in heroes.items():
        all_trajs.append((f"MIMIC-{hero_name}", f"{hero_name.replace('hero_', '').title()}", traj))
        
    # 2. Add 2 Deteriorating cases (but not critical yet)
    deteriorating = [s for s in sim_trajs if s.label_syndrome in ['hemodynamic_instability', 'respiratory_failure', 'cardiac_instability']]
    for i, traj in enumerate(deteriorating[:2]):
        all_trajs.append((f"MIMIC-WARD-{i+10}", f"Patient {i+10} ({traj.label_syndrome[:4].title()})", traj))
        
    # 3. Fill the rest of the ward with stable cases so we have 10-12 total
    stable = [s for s in sim_trajs if s.label_syndrome == 'stable']
    for i, traj in enumerate(stable[:7]):
        all_trajs.append((f"MIMIC-WARD-{i+20}", f"Patient {i+20} (Stable)", traj))

    dataset = []
    base_time = datetime.utcnow()
    for p_id, p_name, traj in all_trajs:
        records = []
        for minute in range(traj.duration_minutes):
            ts = base_time + timedelta(minutes=minute)
            records.append(VitalSignRecord(
                patient_id=p_id, timestamp=ts,
                heart_rate=round(float(traj.vitals["hr"][minute]), 1),
                spo2=round(float(traj.vitals["spo2"][minute]), 1),
                bp_systolic=round(float(traj.vitals["bp_sys"][minute]), 1),
                bp_diastolic=round(float(traj.vitals["bp_dia"][minute]), 1),
                resp_rate=round(float(traj.vitals["rr"][minute]), 1),
                temperature=round(float(traj.vitals["temp"][minute]), 2),
            ))
        drug_events = []
        for d in (traj.drugs if hasattr(traj, 'drugs') and traj.drugs else []):
            drug_events.append(DrugEvent(
                minute=d["minute"], drug_name=d["drug_name"],
                drug_class=d.get("drug_class"), dose=d.get("dose", 0),
                unit=d.get("unit", "mg")
            ))
        dataset.append(PatientCase(
            patient_id=p_id, name=p_name,
            description=f"Source: {'Real MIMIC' if 'REAL' in p_id else 'MIMIC-IV Template'}",
            records=records, drug_events=drug_events,
            duration_minutes=traj.duration_minutes
        ))

    replay = ReplayService(manager, manager.config)
    replay.load_cases(dataset)

    print(
        f"[Replay] Loaded {len(dataset)} patients, "
        f"max duration: {max((c.duration_minutes for c in dataset), default=0)} min"
    )
    print(f"[Replay] Speed: {speed}x, Loop: {loop_replay}")

    tick = 0
    total_records = 0

    # Wait briefly for server to be fully ready
    await asyncio.sleep(1.0)

    # Fast warmup: pre-load 300+ records per patient for instant entropy
    print(f"[Replay] Running fast warmup (pre-loading entropy windows)...")
    try:
        replay.warmup_all_patients(manager)
        
        # Broadcast initial states after warmup
        for case in replay._cases:
            state = manager.get_patient_state(case.patient_id)
            if state:
                sd = state.model_dump(mode="json")
                # Inject ML fields into initial broadcast
                sd["ml_predictions"] = getattr(state, '_ml_predictions', None)
                sd["fusion"] = getattr(state, '_fusion', None)
                sd["detectors"] = getattr(state, '_detectors', [])
                sd["recommendations"] = getattr(state, '_recommendations', {"interventions": [], "suggested_tests": []})
                await ws.broadcast_patient_update(sd)
        print(f"[Replay] Warmup complete. All patients have entropy data.")
    except Exception as e:
        print(f"[Replay] Warmup error (non-fatal): {e}")

    while True:
        try:
            if not replay.is_running:
                if loop_replay:
                    print(f"[Replay] Dataset complete. Restarting...")
                    replay._current_minute = 0
                    replay._running = True
                    await asyncio.sleep(1.0)
                    continue
                else:
                    print(f"[Replay] Dataset complete. Stopping.")
                    await ws.broadcast_status({
                        "replay_finished": True,
                        "total_ticks": tick,
                    })
                    break

            # Process speed ticks per second
            last_states = {}
            for step in range(speed):
                if not replay.is_running:
                    break
                replay.tick()
                total_records += len(replay._cases)
                for case in replay._cases:
                    if replay._current_minute <= len(case.records):
                        state = manager.get_patient_state(case.patient_id)
                        if state:
                            last_states[case.patient_id] = state

            # Broadcast the latest state for each patient
            for pid, state in last_states.items():
                # Only broadcast if we have actual vital data
                if state.calibrating:
                    continue
                state_dict = state.model_dump(mode="json")
                # Inject ML augmentation fields
                state_dict["ml_predictions"] = getattr(state, '_ml_predictions', None)
                state_dict["fusion"] = getattr(state, '_fusion', None)
                state_dict["detectors"] = getattr(state, '_detectors', [])
                state_dict["recommendations"] = getattr(state, '_recommendations', {"interventions": [], "suggested_tests": []})
                # Ensure vital values are present
                vitals = state_dict.get("vitals", {})
                has_data = any(
                    vitals.get(v, {}).get("value") is not None
                    for v in ["heart_rate", "spo2", "bp_systolic", "resp_rate"]
                )
                if has_data:
                    await ws.broadcast_patient_update(state_dict)
                if (
                    state.alert.active
                    and state.alert.severity.value in ("WARNING", "CRITICAL")
                ):
                    await ws.broadcast_alert({
                        "patient_id": pid,
                        "severity": state.alert.severity.value,
                        "message": state.alert.message,
                        "timestamp": state.timestamp.isoformat(),
                        "drug_masked": state.alert.drug_masked,
                    })

            tick += 1
            if tick % 1 == 0:
                progress = replay.progress * 100
                active_alerts = len(manager.get_active_alerts())
                print(
                    f"[Replay] Tick {tick}: {progress:.0f}% | "
                    f"{ws.client_count} WS clients | "
                    f"{active_alerts} active alerts"
                )
                await ws.broadcast_status({
                    "tick": tick,
                    "progress": round(replay.progress, 3),
                    "active_patients": len(last_states),
                    "active_alerts": active_alerts,
                    "ws_clients": ws.client_count,
                    "total_records_processed": total_records,
                    "messages_per_second": len(last_states),
                })

            await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            print("[Replay] Task cancelled.")
            break
        except Exception as e:
            print(f"[Replay] Error: {e}")
            await asyncio.sleep(2.0)


# ------------------------------------------
# FastAPI Lifespan (startup/shutdown)
# ------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage background replay task lifecycle."""
    auto_replay = os.environ.get(
        "CHRONOS_AUTO_REPLAY", "true"
    ).lower() == "true"

    if auto_replay:
        print("[Server] Starting background replay task...")
        task = asyncio.create_task(replay_loop(app))
        app.state.replay_task = task

        # Start background validation computation
        try:
            from ..analytics.validator import ValidationEngine
            config = load_config()
            validator = ValidationEngine(config)
            validator.start_background_validation()
            logger.info("Background validation engine started")
        except Exception as e:
            logger.warning(f"Could not start validation engine: {e}")
    else:
        print(
            "[Server] Auto-replay disabled. "
            "Use run_replay.py to feed data manually."
        )
        app.state.replay_task = None

    yield

    # Shutdown
    if app.state.replay_task and not app.state.replay_task.done():
        app.state.replay_task.cancel()
        try:
            await app.state.replay_task
        except asyncio.CancelledError:
            pass
    print("[Server] Shutdown complete.")


# ------------------------------------------
# App Factory
# ------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = load_config()
    manager = PatientManager(config)
    ws_manager = ConnectionManager()

    app = FastAPI(
        title="Project Chronos",
        description=(
            "Entropy-based ICU Early Warning System. "
            "Real-time entropy analysis with drug awareness "
            "and evidence-based interventions."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Set state before anything else
    app.state.manager = manager
    app.state.ws_manager = ws_manager

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # REST routes
    router = create_router()
    app.include_router(router, prefix="/api/v1")

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """
        Main WebSocket endpoint for real-time updates.

        Clients connect here to receive:
          - patient_update events
          - new_alert events
          - system_status events
        """
        await ws_manager.connect(websocket)
        try:
            # Send full patient states with ML augmentation
            full_states = []
            for pid in manager.latest_states:
                state = manager.get_patient_state(pid)
                if state and not state.calibrating:
                    sd = state.model_dump(mode="json")
                    # Inject ML augmentation fields
                    sd["ml_predictions"] = getattr(state, '_ml_predictions', None)
                    sd["fusion"] = getattr(state, '_fusion', None)
                    sd["detectors"] = getattr(state, '_detectors', [])
                    sd["recommendations"] = getattr(state, '_recommendations', {"interventions": [], "suggested_tests": []})
                    full_states.append(sd)
            
            if full_states:
                # Send as individual patient_update events so frontend processes them correctly
                for state_dict in full_states:
                    await websocket.send_json({
                        "event": "patient_update",
                        "data": state_dict,
                    })
            else:
                summaries = manager.get_all_summaries()
                if summaries:
                    await websocket.send_json({
                        "event": "initial_state",
                        "data": [
                            s.model_dump(mode="json") for s in summaries
                        ],
                    })
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(), timeout=30.0
                    )
                    if data == "ping":
                        await websocket.send_json({"event": "pong"})
                except asyncio.TimeoutError:
                    await websocket.send_json({"event": "keepalive"})
        except WebSocketDisconnect:
            await ws_manager.disconnect(websocket)
        except Exception:
            await ws_manager.disconnect(websocket)

    @app.get("/")
    def root():
        return {
            "name": "Project Chronos",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
            "websocket": "/ws",
            "api": "/api/v1",
        }

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
