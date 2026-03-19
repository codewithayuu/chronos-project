# 🏥 Project Chronos — Entropy-Based ICU Early Warning System

> Replace the binary alarm paradigm with continuous, entropy-aware intelligence
> that tells clinicians not just WHAT a vital sign is, but HOW it is behaving.

## Quick Start

### Option A: Docker (Recommended)

```bash
docker-compose up --build
```

Open:
- **API Docs**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8000/ws
- **Health**: http://localhost:8000/api/v1/system/health

### Option B: Local Python

```bash
pip install -r requirements.txt
python run_server.py
```

## Architecture

```
Bedside Monitor → Ingestion → Entropy Engine → Drug Filter → Evidence Engine → Dashboard
    (MIMIC-IV)     (FastAPI)    (SampEn/MSE)   (15 drugs)    (KNN/500 cases)   (React)
```

### Three Intelligence Layers

| Layer | Name | What It Does |
|-------|------|-------------|
| 1 | **Entropy Engine** | Computes Sample Entropy on vital signs. Healthy = complex. Dying = rigid. |
| 2 | **Drug Filter** | Prevents false alarms from expected drug effects. Detects drug masking. |
| 3 | **Evidence Engine** | Matches patient to 500 historical cases. Ranks interventions by success rate. |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/patients` | Ward view — all patients with CES scores |
| GET | `/api/v1/patients/{id}` | Full patient state with entropy details |
| GET | `/api/v1/patients/{id}/history` | Historical states (for charts) |
| POST | `/api/v1/vitals` | Ingest a vital sign record |
| POST | `/api/v1/patients/{id}/drugs` | Record drug administration |
| GET | `/api/v1/alerts` | All alerts across all patients |
| POST | `/api/v1/alerts/{id}/acknowledge` | Acknowledge an alert |
| GET | `/api/v1/system/health` | System health check |
| WS | `/ws` | Real-time patient updates |

## WebSocket Events

Connect to `ws://localhost:8000/ws` to receive:

```json
{"event": "patient_update", "data": {"patient_id": "...", "composite_entropy": 0.42, ...}}
{"event": "new_alert", "data": {"severity": "WARNING", "message": "...", ...}}
{"event": "system_status", "data": {"tick": 30, "progress": 0.45, ...}}
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHRONOS_SPEED` | 60 | Replay speed (60 = 1 hour of ICU data per minute) |
| `CHRONOS_LOOP` | true | Restart replay when dataset ends |
| `CHRONOS_AUTO_REPLAY` | true | Start replay automatically on server start |
| `CHRONOS_PORT` | 8000 | Server port (docker-compose) |

## Demo Patients

| Patient ID | Scenario | Key Moment |
|------------|----------|------------|
| HERO-SEPSIS-001 | Silent sepsis | Entropy drops at minute ~330 while vitals look normal |
| HERO-RESP-002 | Masked respiratory failure | Propofol hides declining RR entropy |
| HERO-CARD-003 | Cardiac decompensation | HR stays 65-70 (normal range!) but entropy collapses |
| STABLE-001..005 | Healthy controls | High entropy, no alerts |

## Tests

```bash
python -m tests.test_phase1    # Entropy engine (20 tests)
python -m tests.test_phase2    # Drug filter + evidence (15 tests)
python -m tests.test_phase3    # Data pipeline + REST API (20 tests)
python -m tests.test_phase4    # WebSocket + integration (12 tests)
```

## Tech Stack

- **Backend**: Python, FastAPI, NumPy, SciPy
- **Entropy**: Sample Entropy (SampEn), Multi-Scale Entropy (MSE)
- **ML**: KD-Tree KNN (scikit-learn/scipy)
- **Real-time**: WebSockets, async background tasks
- **Deploy**: Docker, docker-compose
