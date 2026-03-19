# ──────────────────────────────────────────────
# Project Chronos — Dockerfile
# Multi-stage build for minimal image size
# ──────────────────────────────────────────────

FROM python:3.11-slim AS base

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.yml .
COPY run_server.py .
COPY run_replay.py .
COPY app/ ./app/
COPY data/ ./data/

# Environment defaults
ENV CHRONOS_SPEED=60
ENV CHRONOS_LOOP=true
ENV CHRONOS_AUTO_REPLAY=true
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/system/health || exit 1

CMD ["python", "run_server.py", "--port", "8000"]
