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

    print(f"🔄 Preparing dataset from MIMIC-IV resources...")
    
    from app.data.synthetic_generator import MIMICIngester, generate_trajectory_simulations, select_hero_cases
    from app.models import VitalSignRecord
    from app.data.generator import PatientCase, DrugEvent
    from datetime import datetime, timedelta
    import numpy as np
    
    rng = np.random.RandomState(42)
    
    # Check for real MIMIC data
    ingester = MIMICIngester("project-chronos/data/mimic-iv-demo")
    real_trajs = ingester.ingest()
    
    # Also generate simulations for the hero cases (ensures we show clear deterioration examples)
    sim_trajs = generate_trajectory_simulations(patients_per_template=10, rng=rng)
    
    # Select the most distinct "Hero" cases from simulations to highlight ML features
    heroes = select_hero_cases(sim_trajs)
    
    # Combine real data with heroes for a complete ward view
    all_trajs = []
    
    # Start with heroes (so bed 01, 02, 03 are the interesting ones)
    for hero_name, traj in heroes.items():
        all_trajs.append((f"MIMIC-{hero_name}", f"{hero_name.replace('hero_', '').title()} Case", traj))
        
    # Add real data points if any exist
    for i, traj in enumerate(real_trajs[:10]): # Limit to first 10 real patients
        all_trajs.append((f"MIMIC-REAL-{traj.subject_id}", f"MIMIC Patient {traj.subject_id}", traj))
        
    # If no real data, fill out with simulations
    if not real_trajs:
        stable = [s for s in sim_trajs if s.label_syndrome == 'stable'][:7]
        for i, traj in enumerate(stable):
            all_trajs.append((f"MIMIC-SIM-{i}", f"Simulated Patient {i}", traj))

    # Convert to PatientCase format
    dataset = []
    base_time = datetime.utcnow()
    
    for p_id, p_name, traj in all_trajs:
        records = []
        for minute in range(traj.duration_minutes):
            ts = base_time + timedelta(minutes=minute)
            records.append(VitalSignRecord(
                patient_id=p_id,
                timestamp=ts,
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
                minute=d["minute"],
                drug_name=d["drug_name"],
                drug_class=d.get("drug_class"),
                dose=d.get("dose", 0),
                unit=d.get("unit", "mg")
            ))
            
        dataset.append(PatientCase(
            patient_id=p_id,
            name=p_name,
            description=f"Source: {'Real MIMIC' if 'REAL' in p_id else 'MIMIC-IV Template'}",
            records=records,
            drug_events=drug_events,
            duration_minutes=traj.duration_minutes
        ))

    # Create a local pipeline and load cases
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
