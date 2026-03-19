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




---


# Project Chronos — ICU Entropy Monitoring Dashboard

Real-time ICU monitoring dashboard that visualizes patient deterioration through entropy analysis of physiological signals. Built for the detection of clinical decline hours before traditional alarm systems trigger.

## Architecture

```
Frontend (React)                    Backend (Python/FastAPI)
Port 3000                          Port 8000
    |                                  |
    |--- WebSocket (ws://..../ws) -----|  Real-time patient updates
    |--- REST API (http://.../api) ----|  Historical data, alerts
    |                                  |
    v                                  v
 Ward View (8 patients)          Simulation Engine
 Patient Detail View             Entropy Calculator
 Alert Feed                      Evidence Engine
 CES Gauge + Charts              Drug Interaction Model
```

## Quick Start

### 1. Start the Backend

```bash
cd project-chronos
pip install -r requirements.txt
python run_server.py
```

Backend runs at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### 2. Start the Frontend

```bash
cd chronos-dashboard
npm install
npm start
```

Dashboard opens at `http://localhost:3000`.

## Features

### Ward View
- Grid of patient cards auto-sorted by severity (CRITICAL first)
- Real-time CES scores with animated number transitions
- Enhanced sparklines with gradient fills and breathing current-value indicators
- Severity color system with pulsing glow animations
- Calibration overlay with progress bar for new patients
- System status bar showing connection health and data throughput

### Patient Detail View
- Five dual-axis vital sign charts (Heart Rate, SpO2, BP, Resp Rate, Temperature)
- Entropy overlay on each chart showing complexity loss
- Traditional alarm threshold lines for visual contrast
- Drug event markers showing medication timing
- Contributing vital highlighting when entropy is flagging specific signals
- CES gauge with arc animation and raw vs adjusted comparison
- CES history trend chart showing entropy trajectory over time
- Per-vital entropy bars with individual severity coloring
- Ranked intervention cards with success rates and evidence sources
- Clinical disclaimer for decision support context

### Alert System
- Slide-in alert panel with active and acknowledged sections
- Real-time alert streaming via WebSocket
- One-click acknowledge with attribution
- Keyboard accessible with focus trapping

### Animations
- Framer Motion page transitions with blur effects
- Spring physics on all interactive elements
- Staggered grid and list entrances
- Smooth card reordering on severity changes
- CES gauge arc draw animation
- Animated number counting for all metrics

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | React 18 |
| Routing | React Router v6 |
| Charts | Recharts |
| Animation | Framer Motion |
| Icons | Phosphor Icons |
| Styling | CSS Custom Properties |
| Data | WebSocket + REST API |
| Fonts | Plus Jakarta Sans, JetBrains Mono |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REACT_APP_API_URL` | `http://localhost:8000` | Backend API base URL |
| `REACT_APP_WS_URL` | `ws://localhost:8000/ws` | WebSocket endpoint |

## Project Structure

```
src/
├── index.js                    # Entry point with ErrorBoundary
├── index.css                   # Global styles + accessibility
├── App.js                      # Root component with routing
├── App.css                     # App shell styles
├── hooks/
│   ├── useChronosWebSocket.js  # WebSocket connection + state
│   ├── usePatientHistory.js    # REST API history fetching
│   ├── useAlerts.js            # REST API alert fetching
│   └── useSparklineBuffer.js   # Sliding window data buffer
├── components/
│   ├── Header.jsx/.css         # Navigation + status bar
│   ├── WardView.jsx/.css       # Patient grid (main view)
│   ├── PatientCard.jsx/.css    # Individual patient card
│   ├── PatientDetailView.jsx/css # Full patient detail
│   ├── VitalChart.jsx/.css     # Dual-axis vital sign chart
│   ├── CESGauge.jsx/.css       # SVG arc gauge component
│   ├── CESHistoryChart.jsx/css # CES trend area chart
│   ├── EntropyBars.jsx/.css    # Per-vital entropy bars
│   ├── AlertBanner.jsx/.css    # Severity alert banner
│   ├── AlertFeed.jsx/.css      # Slide-in alert panel
│   ├── InterventionCard.jsx/css# Ranked intervention card
│   ├── DrugBadge.jsx/.css      # Medication display
│   ├── EnhancedSparkline.jsx   # SVG sparkline with effects
│   ├── Sparkline.jsx           # Basic SVG sparkline
│   ├── AnimatedNumber.jsx      # Smooth number transitions
│   ├── LiveDot.jsx             # Breathing status indicator
│   ├── DataFlowIndicator.jsx/css # Message rate display
│   ├── SystemStatusBar.jsx/css # System health footer
│   ├── ConnectionStatus.jsx    # Connection state display
│   ├── ErrorBoundary.jsx/.css  # Error catch + recovery
│   ├── InitialLoader.jsx/.css  # Boot loading screen
│   └── SkipToContent.jsx/.css  # Accessibility skip link
├── utils/
│   ├── constants.js            # Severity config, thresholds
│   ├── helpers.js              # Formatting, sorting
│   ├── animations.js           # Motion variants + springs
│   └── chartHelpers.js         # Chart color + domain logic
└── styles/
    └── variables.css           # CSS custom properties
```

## Accessibility

- Skip-to-content link for keyboard users
- ARIA labels on all interactive elements
- Focus trapping in modal dialogs
- `prefers-reduced-motion` support disables all animations
- `forced-colors` mode support for high contrast
- Color is never the sole indicator (always paired with text/icons)
- Semantic HTML structure with proper heading hierarchy
- Screen reader announcements for connection status changes

## Performance Notes

- All animations use `transform` and `opacity` only (GPU composited)
- Noise overlay applied to fixed pseudo-element (no scroll repaint)
- Sparklines use pure SVG (no DOM thrashing)
- Chart data downsampled for large datasets
- WebSocket reconnection with exponential backoff
- React.memo on all leaf components
- AnimatePresence prevents unmounted component animation

## API Reference

See the backend `README.md` or visit `http://localhost:8000/docs` for the complete interactive API documentation.

## License

Hackathon project. Internal use only.
