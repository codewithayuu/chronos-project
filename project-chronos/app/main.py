"""
Project Chronos — FastAPI Application Entry Point.

Start with:
    python -m app.main
    or:
    uvicorn app.main:app --host 0.0.0.0 --port8000 --reload
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import load_config
from .pipeline import ChronosPipeline
from .data.replay import ReplayService
from .api.routes import create_router
from .analytics.validator import ValidationEngine

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    On startup: initialize pipeline, load data, start replay service.
    On shutdown: stop replay service gracefully.
    """
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("  PROJECT CHRONOS — Starting Up")
    print("=" * 60)

    config = load_config()

    # Initialize the processing pipeline
    pipeline = ChronosPipeline(config)
    app.state.pipeline = pipeline

    # Initialize and start replay service
    replay = ReplayService(pipeline, config)
    replay.load_cases()
    app.state.replay = replay

    # Start background validation
    try:
        validator = ValidationEngine(pipeline.config if hasattr(pipeline, 'config') else None)
        validator.start_background_validation()
        logger.info("Background validation engine started")
    except Exception as e:
        logger.warning(f"Could not start validation engine: {e}")

    # Start replay in background
    replay_task = asyncio.create_task(replay.run())

    print("=" * 60)
    print(f"  API:       http://localhost:{config.api.port}/api/v1")
    print(f"  Health:    http://localhost:{config.api.port}/api/v1/system/health")
    print(f"  Patients:  http://localhost:{config.api.port}/api/v1/patients")
    print(f"  Docs:      http://localhost:{config.api.port}/docs")
    print(f"  Replay:    {config.data_replay.speed_multiplier}x speed")
    print("=" * 60)

    yield

    # Shutdown
    print("\n[Chronos] Shutting down...")
    replay.stop()
    replay_task.cancel()
    try:
        await replay_task
    except asyncio.CancelledError:
        pass
    print("[Chronos] Goodbye.")

# ──────────────────────────────────────────────


# Create the FastAPI application


# ──────────────────────────────────────────────

app = FastAPI(
    title="Project Chronos",
    description="Entropy-based ICU Early Warning System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permissive for hackathon demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
router = create_router()
app.include_router(router)

# ──────────────────────────────────────────────


# Direct execution support


# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    config = load_config()
    uvicorn.run(
        "app.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=False,
        log_level="info",
    )
