"""
Data Replay Script — feeds synthetic patient data into running API.

Usage:
    1. Start server:  python run_server.py
    2. In another terminal: python run_replay.py

Options:
    --speed   Speed multiplier (default: 60 = one simulated minute per real second)
    --url     API base URL (default: http://localhost:8000)
"""

import argparse
import time
import sys
import httpx

from app.data.generator import DataGenerator
from app.data.replay import ReplayService
from app.core.manager import PatientManager
from app.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Project Chronos Data Replay")
    parser.add_argument("--speed", type=int, default=60, help="Speed multiplier")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    delay = 1.0 / args.speed if args.speed > 0 else 1.0
    api_url = f"{args.url}/api/v1"

    print(f"🔄 Generating demo dataset...")
    dataset = DataGenerator.generate_demo_dataset()
    
    # Create a local pipeline for replay service
    config = load_config()
    pipeline = PatientManager(config)
    replay = ReplayService(pipeline, config)
    replay.load_cases(dataset)

    print(f"📊 Dataset: {len(replay._cases)} cases loaded")
    print(f"⏱️  Speed: {args.speed}x (1 record every {delay:.3f}s)")
    print(f"🌐 Target: {api_url}")
    print(f"{'─' * 50}")

    # Check server is up
    try:
        r = httpx.get(f"{api_url}/system/health", timeout=3.0)
        r.raise_for_status()
        print(f"✅ Server is up: {r.json()}")
    except Exception as e:
        print(f"❌ Cannot reach server at {api_url}: {e}")
        print(f"   Start the server first: python run_server.py")
        sys.exit(1)

    print(f"\n🚀 Starting replay...\n")

    try:
        # Run the replay service async loop
        import asyncio
        asyncio.run(replay.run())
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Replay stopped by user")


if __name__ == "__main__":
    main()
