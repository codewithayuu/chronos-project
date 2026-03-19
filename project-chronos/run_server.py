"""
Start the Project Chronos API server with background replay.

Usage:
    python run_server.py                  # Default: port 8000, auto-replay ON
    python run_server.py --port 8001      # Custom port
    python run_server.py --no-replay      # Disable auto-replay (use run_replay.py)

Environment variables:
    CHRONOS_SPEED=60         Replay speed multiplier (default: 60)
    CHRONOS_LOOP=true        Loop replay when finished (default: true)
    CHRONOS_AUTO_REPLAY=true Start replay automatically (default: true)
"""

import argparse
import os
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Project Chronos API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--no-replay", action="store_true",
                        help="Disable automatic background replay")
    parser.add_argument("--speed", type=int, default=60,
                        help="Replay speed multiplier")
    args = parser.parse_args()

    if args.no_replay:
        os.environ["CHRONOS_AUTO_REPLAY"] = "false"
    os.environ.setdefault("CHRONOS_SPEED", str(args.speed))

    print(f"🏥 Project Chronos — ICU Early Warning System")
    print(f"   API:       http://localhost:{args.port}/api/v1")
    print(f"   Docs:      http://localhost:{args.port}/docs")
    print(f"   WebSocket: ws://localhost:{args.port}/ws")
    print(f"   Speed:     {args.speed}x")
    print(f"   Replay:    {'AUTO' if not args.no_replay else 'MANUAL'}")
    print()

    uvicorn.run(
        "app.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
