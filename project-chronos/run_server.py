
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
