"""
Synthetic Data Generator for ML Training.

Reads MIMIC-IV Demo tables (chartevents, icustays, inputevents, patients, admissions)
and:
  1. Extracts real ICU vital-sign trajectories
  2. Creates statistical clones (noise-augmented copies)
  3. Generates parametric trajectory simulations across 5 deterioration templates
  4. Labels deterioration (1h, 4h, 8h) and syndrome outcomes
  5. Outputs numpy arrays + metadata for training scripts

Schema mapping:
  chartevents.csv.gz → vitals time series (HR, BP, RR, SpO2, Temp)
  icustays.csv.gz    → ICU stay boundaries
  inputevents.csv.gz → drug administration events
  patients.csv.gz    → age, sex
  admissions.csv.gz  → admission/discharge times, mortality
"""

import gzip
import csv
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════
# MIMIC-IV Item ID → Vital Name Mapping
# ═══════════════════════════════════════════════════

# Official MIMIC-IV chartevents item IDs for vitals
VITAL_ITEM_IDS = {
    # Heart Rate
    220045: "hr",
    # Systolic BP (non-invasive)
    220179: "bp_sys",
    # Diastolic BP (non-invasive)
    220180: "bp_dia",
    # Systolic BP (arterial line)
    220050: "bp_sys",
    # Diastolic BP (arterial line)
    220051: "bp_dia",
    # Respiratory Rate
    220210: "rr",
    # SpO2
    220277: "spo2",
    # Temperature (C)
    223761: "temp",
    # Temperature (F → will convert)
    223762: "temp_f",
}

# Drug class mapping for inputevents
DRUG_CLASS_MAP = {
    "norepinephrine": "vasopressor",
    "vasopressin": "vasopressor",
    "dopamine": "vasopressor",
    "phenylephrine": "vasopressor",
    "epinephrine": "vasopressor",
    "propofol": "sedative",
    "midazolam": "sedative",
    "dexmedetomidine": "sedative",
    "fentanyl": "opioid",
    "morphine": "opioid",
    "hydromorphone": "opioid",
    "metoprolol": "beta_blocker",
    "esmolol": "beta_blocker",
    "labetalol": "beta_blocker",
    "dobutamine": "inotrope",
    "milrinone": "inotrope",
    "furosemide": "diuretic",
}


@dataclass
class PatientTrajectory:
    """A complete vital trajectory for one ICU stay."""
    subject_id: int
    stay_id: int
    age: float
    sex: str
    vitals: Dict[str, np.ndarray]  # {vital_name: array of values, 1 per minute}
    drugs: List[Dict]              # [{drug_name, drug_class, minute, dose, unit}]
    duration_minutes: int
    died: bool = False
    label_syndrome: str = "stable"


# ═══════════════════════════════════════════════════
# MIMIC-IV Demo Ingestion
# ═══════════════════════════════════════════════════

class MIMICIngester:
    """Reads MIMIC-IV Demo csvs and extracts ICU trajectories."""

    def __init__(self, mimic_dir: str = "data/mimic-iv-demo"):
        self.mimic_dir = Path(mimic_dir)
        self.icu_dir = self.mimic_dir / "icu"
        self.hosp_dir = self.mimic_dir / "hosp"

    def _read_gz_csv(self, path: Path) -> pd.DataFrame:
        """Read a gzipped CSV file."""
        return pd.read_csv(path, compression="gzip", low_memory=False)

    def available(self) -> bool:
        """Check if MIMIC-IV demo files are available."""
        return (self.icu_dir / "chartevents.csv.gz").exists()

    def ingest(self) -> List[PatientTrajectory]:
        """
        Full MIMIC-IV Demo ingestion pipeline.

        Returns list of PatientTrajectory objects with minute-resolution
        vital signs, drug events, and labels.
        """
        if not self.available():
            print("[MIMIC] Demo files not found. Using synthetic-only mode.")
            return []

        print("[MIMIC] Loading ICU stays...")
        stays = self._read_gz_csv(self.icu_dir / "icustays.csv.gz")

        print("[MIMIC] Loading patients...")
        patients = self._read_gz_csv(self.hosp_dir / "patients.csv.gz")

        print("[MIMIC] Loading admissions...")
        admissions = self._read_gz_csv(self.hosp_dir / "admissions.csv.gz")

        print("[MIMIC] Loading chart events (vitals)... this may take a moment")
        chartevents = self._read_gz_csv(self.icu_dir / "chartevents.csv.gz")

        print("[MIMIC] Loading input events (drugs)...")
        try:
            inputevents = self._read_gz_csv(self.icu_dir / "inputevents.csv.gz")
        except Exception:
            inputevents = pd.DataFrame()

        # Build trajectories
        trajectories = []

        # join patients + admissions + stays
        stays = stays.merge(patients, on="subject_id", how="left")
        stays = stays.merge(
            admissions[["hadm_id", "deathtime", "hospital_expire_flag"]],
            on="hadm_id",
            how="left",
        )

        # Parse timestamps
        stays["intime"] = pd.to_datetime(stays["intime"])
        stays["outtime"] = pd.to_datetime(stays["outtime"])

        # Filter chartevents to vital signs only
        vital_ids = set(VITAL_ITEM_IDS.keys())
        chartevents = chartevents[chartevents["itemid"].isin(vital_ids)].copy()
        chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])
        chartevents["valuenum"] = pd.to_numeric(
            chartevents["valuenum"], errors="coerce"
        )
        chartevents = chartevents.dropna(subset=["valuenum"])

        # Group by stay
        chart_by_stay = chartevents.groupby("stay_id")

        for _, stay_row in stays.iterrows():
            stay_id = stay_row["stay_id"]
            subject_id = stay_row["subject_id"]
            intime = stay_row["intime"]
            outtime = stay_row["outtime"]
            duration = (outtime - intime).total_seconds() / 60.0

            if duration < 120 or duration > 1440 * 7:
                continue  # skip too short or too long stays

            duration_minutes = int(min(duration, 720))  # cap at 12 hours

            # Age
            age = float(stay_row.get("anchor_age", 60))
            sex = str(stay_row.get("gender", "M"))

            # Mortality
            died = bool(stay_row.get("hospital_expire_flag", 0))

            # Extract vitals for this stay
            vitals = self._extract_vitals(
                chart_by_stay, stay_id, intime, duration_minutes
            )

            if vitals is None:
                continue  # not enough vital data

            # Extract drugs
            drugs = self._extract_drugs(
                inputevents, stay_id, intime, duration_minutes
            )

            # Assign syndrome label based on vital patterns
            syndrome = self._infer_syndrome(vitals, died)

            trajectories.append(
                PatientTrajectory(
                    subject_id=subject_id,
                    stay_id=stay_id,
                    age=age,
                    sex=sex,
                    vitals=vitals,
                    drugs=drugs,
                    duration_minutes=duration_minutes,
                    died=died,
                    label_syndrome=syndrome,
                )
            )

        print(f"[MIMIC] Extracted {len(trajectories)} ICU trajectories")
        return trajectories

    def _extract_vitals(
        self, chart_by_stay, stay_id, intime, duration_minutes
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract minute-resolution vitals for a stay."""
        try:
            events = chart_by_stay.get_group(stay_id)
        except KeyError:
            return None

        vitals = {v: np.full(duration_minutes, np.nan) for v in ["hr", "bp_sys", "bp_dia", "rr", "spo2", "temp"]}

        for _, row in events.iterrows():
            charttime = row["charttime"]
            item_id = row["itemid"]
            value = row["valuenum"]

            minute = int((charttime - intime).total_seconds() / 60.0)
            if minute < 0 or minute >= duration_minutes:
                continue

            vital_name = VITAL_ITEM_IDS.get(item_id)
            if vital_name == "temp_f":
                value = (value - 32.0) * 5.0 / 9.0  # F → C
                vital_name = "temp"

            if vital_name and vital_name in vitals:
                # Validate physiological range
                if self._validate_vital(vital_name, value):
                    vitals[vital_name][minute] = value

        # Forward-fill gaps (vitals are typically charted every 5-60 min)
        for name in vitals:
            vitals[name] = self._forward_fill(vitals[name])

        # Check if we have enough data
        hr_coverage = np.sum(~np.isnan(vitals["hr"])) / duration_minutes
        if hr_coverage < 0.3:
            return None  # insufficient data

        # Fill remaining NaN with defaults
        defaults = {"hr": 80, "bp_sys": 120, "bp_dia": 75, "rr": 16, "spo2": 97, "temp": 37.0}
        for name, default in defaults.items():
            vitals[name] = np.nan_to_num(vitals[name], nan=default)

        return vitals

    def _extract_drugs(
        self, inputevents, stay_id, intime, duration_minutes
    ) -> List[Dict]:
        """Extract drug administration events for a stay."""
        drugs = []
        if inputevents.empty:
            return drugs

        try:
            stay_events = inputevents[inputevents["stay_id"] == stay_id]
        except Exception:
            return drugs

        for _, row in stay_events.iterrows():
            label = str(row.get("label", row.get("ordercategorydescription", ""))).lower()
            starttime = pd.to_datetime(row.get("starttime"))

            drug_class = None
            drug_name = label
            for dname, dclass in DRUG_CLASS_MAP.items():
                if dname in label:
                    drug_name = dname.title()
                    drug_class = dclass
                    break

            if drug_class is None:
                continue  # skip non-matched drugs

            minute = int((starttime - intime).total_seconds() / 60.0)
            if minute < 0 or minute >= duration_minutes:
                continue

            dose = float(row.get("amount", row.get("rate", 0)) or 0)
            unit = str(row.get("amountuom", row.get("rateuom", "mg")) or "mg")

            drugs.append({
                "drug_name": drug_name,
                "drug_class": drug_class,
                "minute": minute,
                "dose": dose,
                "unit": unit,
            })

        return drugs

    @staticmethod
    def _validate_vital(name: str, value: float) -> bool:
        """Check if a vital-sign value is within physiological range."""
        ranges = {
            "hr": (20, 250),
            "bp_sys": (30, 300),
            "bp_dia": (10, 200),
            "rr": (2, 60),
            "spo2": (40, 100),
            "temp": (30, 43),
        }
        lo, hi = ranges.get(name, (0, 1e6))
        return lo <= value <= hi

    @staticmethod
    def _forward_fill(arr: np.ndarray) -> np.ndarray:
        """Forward-fill NaN values in an array."""
        result = arr.copy()
        last_valid = np.nan
        for i in range(len(result)):
            if np.isnan(result[i]):
                if not np.isnan(last_valid):
                    result[i] = last_valid
            else:
                last_valid = result[i]
        return result

    @staticmethod
    def _infer_syndrome(vitals: Dict[str, np.ndarray], died: bool) -> str:
        """
        Infer deterioration syndrome from vital patterns.
        Simple heuristic-based labeling.
        """
        hr = vitals["hr"]
        bp = vitals["bp_sys"]
        rr = vitals["rr"]
        spo2 = vitals["spo2"]
        temp = vitals["temp"]

        # Use last quarter of the trajectory for labeling
        n = len(hr)
        q4_start = int(n * 0.75)
        hr_q4 = hr[q4_start:]
        bp_q4 = bp[q4_start:]
        rr_q4 = rr[q4_start:]
        spo2_q4 = spo2[q4_start:]
        temp_q4 = temp[q4_start:]

        mean_hr = np.mean(hr_q4)
        mean_bp = np.mean(bp_q4)
        mean_rr = np.mean(rr_q4)
        mean_spo2 = np.mean(spo2_q4)

        # Sepsis-like: tachycardia + hypotension + fever + tachypnea
        if mean_hr > 100 and mean_bp < 90 and np.mean(temp_q4) > 38.0:
            return "sepsis_like"

        # Respiratory failure: low SpO2 + high RR
        if mean_spo2 < 92 and mean_rr > 24:
            return "respiratory_failure"

        # Hemodynamic instability: hypotension without sepsis
        if mean_bp < 85 and mean_hr > 90:
            return "hemodynamic_instability"

        # Cardiac instability: extreme HR changes
        if mean_hr > 120 or mean_hr < 50:
            return "cardiac_instability"

        return "stable"


# ═══════════════════════════════════════════════════
# Statistical Cloning
# ═══════════════════════════════════════════════════

def create_statistical_clones(
    trajectories: List[PatientTrajectory],
    clones_per_patient: int = 5,
    noise_fraction: float = 0.25,
    offset_fraction: float = 0.30,
    rng: np.random.RandomState = None,
) -> List[PatientTrajectory]:
    """
    Clone each trajectory by adding realistic noise.

    Uses 25% of SD for noise and 30% for offset (not 10%/15% as orig PRD,
    which produced near-identical copies).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    clones = []
    for traj in trajectories:
        for c in range(clones_per_patient):
            cloned_vitals = {}
            for vital_name, arr in traj.vitals.items():
                sd = np.std(arr) if len(arr) > 1 else 1.0
                noise = rng.randn(len(arr)) * sd * noise_fraction
                offset = rng.randn() * sd * offset_fraction
                cloned = arr + noise + offset

                # Clamp to physiological ranges
                cloned = _clamp_vital(vital_name, cloned)
                cloned_vitals[vital_name] = cloned

            clone = PatientTrajectory(
                subject_id=traj.subject_id,
                stay_id=traj.stay_id * 1000 + c,
                age=max(18, traj.age + rng.randn() * 3),
                sex=traj.sex,
                vitals=cloned_vitals,
                drugs=traj.drugs.copy(),
                duration_minutes=traj.duration_minutes,
                died=traj.died,
                label_syndrome=traj.label_syndrome,
            )
            clones.append(clone)

    return clones


# ═══════════════════════════════════════════════════
# Trajectory Simulation Templates
# ═══════════════════════════════════════════════════

TRAJECTORY_TEMPLATES = {
    "sepsis": {
        "syndrome": "sepsis_like",
        "phases": [
            {"minutes": 300, "hr": 78, "bp_sys": 120, "rr": 15, "spo2": 97, "temp": 37.0, "variability": 1.0},
            {"minutes": 120, "hr": 90, "bp_sys": 105, "rr": 20, "spo2": 95, "temp": 37.8, "variability": 0.4},
            {"minutes": 120, "hr": 108, "bp_sys": 85, "rr": 28, "spo2": 92, "temp": 38.5, "variability": 0.15},
            {"minutes": 180, "hr": 120, "bp_sys": 72, "rr": 32, "spo2": 88, "temp": 39.0, "variability": 0.05},
        ],
        "drugs": [{"drug_name": "Norepinephrine", "drug_class": "vasopressor", "relative_minute_fraction": 0.65, "dose": 0.08, "unit": "mcg/kg/min"}],
        "mortality_rate": 0.25,
    },
    "respiratory": {
        "syndrome": "respiratory_failure",
        "phases": [
            {"minutes": 300, "hr": 82, "bp_sys": 125, "rr": 16, "spo2": 98, "temp": 36.9, "variability": 1.0},
            {"minutes": 120, "hr": 88, "bp_sys": 118, "rr": 22, "spo2": 95, "temp": 37.0, "variability": 0.5},
            {"minutes": 120, "hr": 95, "bp_sys": 112, "rr": 30, "spo2": 90, "temp": 37.1, "variability": 0.15},
            {"minutes": 180, "hr": 105, "bp_sys": 108, "rr": 36, "spo2": 84, "temp": 37.2, "variability": 0.05},
        ],
        "drugs": [],
        "mortality_rate": 0.20,
    },
    "hemodynamic": {
        "syndrome": "hemodynamic_instability",
        "phases": [
            {"minutes": 300, "hr": 76, "bp_sys": 128, "rr": 14, "spo2": 97, "temp": 37.0, "variability": 1.0},
            {"minutes": 150, "hr": 85, "bp_sys": 100, "rr": 18, "spo2": 96, "temp": 37.1, "variability": 0.4},
            {"minutes": 120, "hr": 100, "bp_sys": 78, "rr": 22, "spo2": 93, "temp": 37.2, "variability": 0.12},
            {"minutes": 150, "hr": 118, "bp_sys": 65, "rr": 28, "spo2": 90, "temp": 37.3, "variability": 0.04},
        ],
        "drugs": [{"drug_name": "Norepinephrine", "drug_class": "vasopressor", "relative_minute_fraction": 0.55, "dose": 0.1, "unit": "mcg/kg/min"}],
        "mortality_rate": 0.30,
    },
    "cardiac": {
        "syndrome": "cardiac_instability",
        "phases": [
            {"minutes": 300, "hr": 88, "bp_sys": 130, "rr": 16, "spo2": 96, "temp": 37.1, "variability": 0.9},
            {"minutes": 180, "hr": 72, "bp_sys": 118, "rr": 15, "spo2": 95, "temp": 37.0, "variability": 0.3},
            {"minutes": 120, "hr": 58, "bp_sys": 90, "rr": 20, "spo2": 92, "temp": 37.1, "variability": 0.10},
            {"minutes": 120, "hr": 48, "bp_sys": 75, "rr": 26, "spo2": 88, "temp": 37.2, "variability": 0.03},
        ],
        "drugs": [
            {"drug_name": "Metoprolol", "drug_class": "beta_blocker", "relative_minute_fraction": 0.25, "dose": 5.0, "unit": "mg"},
            {"drug_name": "Dobutamine", "drug_class": "inotrope", "relative_minute_fraction": 0.70, "dose": 5.0, "unit": "mcg/kg/min"},
        ],
        "mortality_rate": 0.30,
    },
    "stable": {
        "syndrome": "stable",
        "phases": [
            {"minutes": 720, "hr": 75, "bp_sys": 122, "rr": 15, "spo2": 97, "temp": 37.0, "variability": 0.95},
        ],
        "drugs": [],
        "mortality_rate": 0.02,
    },
}


def generate_trajectory_simulations(
    patients_per_template: int = 40,
    rng: np.random.RandomState = None,
) -> List[PatientTrajectory]:
    """
    Generate 200 simulated trajectories (5 templates × 40 patients each).
    Randomizes phase timing, deterioration speed, and base vital values.
    """
    if rng is None:
        rng = np.random.RandomState(123)

    simulations = []
    counter = 0

    for template_name, template in TRAJECTORY_TEMPLATES.items():
        for i in range(patients_per_template):
            counter += 1

            # Randomize base vitals
            age = max(18, min(95, rng.normal(60, 15)))
            sex = rng.choice(["M", "F"])

            # Randomize phase timing (±2 hours)
            time_jitter = rng.uniform(0.8, 1.2)
            rate_jitter = rng.uniform(0.8, 1.2)

            # Build minute-by-minute vitals
            total_minutes = 720
            vitals = {v: np.zeros(total_minutes) for v in ["hr", "bp_sys", "bp_dia", "rr", "spo2", "temp"]}

            minute_cursor = 0
            for pi, phase in enumerate(template["phases"]):
                phase_duration = int(phase["minutes"] * time_jitter)
                phase_duration = max(30, min(phase_duration, total_minutes - minute_cursor))

                if pi < len(template["phases"]) - 1:
                    next_phase = template["phases"][pi + 1]
                else:
                    next_phase = phase  # last phase holds steady

                for m in range(phase_duration):
                    t = m / max(phase_duration, 1)
                    # Cosine interpolation
                    frac = 0.5 * (1 - np.cos(t * np.pi))

                    abs_minute = minute_cursor + m
                    if abs_minute >= total_minutes:
                        break

                    v = phase["variability"] + frac * (next_phase.get("variability", phase["variability"]) - phase["variability"])

                    for vital_name, vital_key in [("hr", "hr"), ("bp_sys", "bp_sys"), ("rr", "rr"), ("spo2", "spo2"), ("temp", "temp")]:
                        base = phase[vital_name] + frac * (next_phase[vital_name] - phase[vital_name])
                        # Add individual variation
                        base += rng.randn() * _VITAL_BASE_STD.get(vital_name, 1.0) * 0.3
                        noise = rng.randn() * _VITAL_BASE_STD.get(vital_name, 1.0) * v
                        periodic = np.sin((abs_minute + rng.randint(0, 100)) * 0.1) * _VITAL_BASE_STD.get(vital_name, 1.0) * 0.2 * (1 - v)
                        val = base + noise + periodic

                        vitals[vital_name][abs_minute] = val

                    # BP diastolic derived from systolic
                    vitals["bp_dia"][abs_minute] = vitals["bp_sys"][abs_minute] * rng.uniform(0.58, 0.72)

                minute_cursor += phase_duration

            # Fill any remaining zeros with last valid value
            for name in vitals:
                if minute_cursor < total_minutes:
                    vitals[name][minute_cursor:] = vitals[name][minute_cursor - 1]

            # Clamp all vitals
            for name in vitals:
                vitals[name] = _clamp_vital(name, vitals[name])

            # Generate drug events
            drugs = []
            for drug_def in template["drugs"]:
                drug_minute = int(total_minutes * drug_def["relative_minute_fraction"] + rng.randint(-120, 120))
                drug_minute = max(0, min(drug_minute, total_minutes - 1))
                drugs.append({
                    "drug_name": drug_def["drug_name"],
                    "drug_class": drug_def["drug_class"],
                    "minute": drug_minute,
                    "dose": drug_def["dose"] * rng.uniform(0.7, 1.3),
                    "unit": drug_def["unit"],
                })

            died = rng.random() < template["mortality_rate"] * rate_jitter

            simulations.append(
                PatientTrajectory(
                    subject_id=10000 + counter,
                    stay_id=20000 + counter,
                    age=age,
                    sex=sex,
                    vitals=vitals,
                    drugs=drugs,
                    duration_minutes=total_minutes,
                    died=died,
                    label_syndrome=template["syndrome"],
                )
            )

    return simulations


# ═══════════════════════════════════════════════════
# Deterioration Label Generator
# ═══════════════════════════════════════════════════

def generate_deterioration_labels(
    trajectory: PatientTrajectory,
) -> Dict[str, np.ndarray]:
    """
    Generate binary deterioration labels for 1h, 4h, 8h horizons.

    A deterioration event is defined as:
    - HR > 120 or < 45
    - BP_sys < 80
    - SpO2 < 88
    - RR > 35 or < 6
    """
    n = trajectory.duration_minutes
    hr = trajectory.vitals["hr"]
    bp = trajectory.vitals["bp_sys"]
    spo2 = trajectory.vitals["spo2"]
    rr = trajectory.vitals["rr"]

    # Find deterioration event times
    event_mask = np.zeros(n, dtype=bool)
    for t in range(n):
        if (hr[t] > 120 or hr[t] < 45 or
            bp[t] < 80 or
            spo2[t] < 88 or
            rr[t] > 35 or rr[t] < 6):
            event_mask[t] = True

    # For each time point, check if an event occurs within the horizon
    labels = {}
    for horizon_name, horizon_min in [("1h", 60), ("4h", 240), ("8h", 480)]:
        label = np.zeros(n, dtype=int)
        for t in range(n):
            end = min(t + horizon_min, n)
            if np.any(event_mask[t:end]):
                label[t] = 1
        labels[horizon_name] = label

    return labels


# ═══════════════════════════════════════════════════
# Hero Case Selection
# ═══════════════════════════════════════════════════

def select_hero_cases(
    trajectories: List[PatientTrajectory],
) -> Dict[str, PatientTrajectory]:
    """
    Select the best 3 hero cases from simulated trajectories.

    Hero 1 (Sepsis): vitals stay normal longest while entropy-proxy drops
    Hero 2 (Respiratory): RR entropy drops before SpO2 drops
    Hero 3 (Cardiac): dramatic HR entropy changes
    """
    heroes = {}

    sepsis = [t for t in trajectories if t.label_syndrome == "sepsis_like"]
    if sepsis:
        # Pick the one with the longest "normal-looking" phase
        best = max(sepsis, key=lambda t: _count_normal_minutes(t))
        heroes["hero_sepsis"] = best

    resp = [t for t in trajectories if t.label_syndrome == "respiratory_failure"]
    if resp:
        best = max(resp, key=lambda t: _spo2_drop_delay(t))
        heroes["hero_respiratory"] = best

    cardiac = [t for t in trajectories if t.label_syndrome == "cardiac_instability"]
    if cardiac:
        best = max(cardiac, key=lambda t: np.std(t.vitals["hr"]))
        heroes["hero_cardiac"] = best

    return heroes


def _count_normal_minutes(t: PatientTrajectory) -> int:
    """Count minutes where vitals are in normal range."""
    hr, bp, spo2 = t.vitals["hr"], t.vitals["bp_sys"], t.vitals["spo2"]
    normal = 0
    for i in range(len(hr)):
        if 50 < hr[i] < 120 and 90 < bp[i] < 180 and spo2[i] > 90:
            normal += 1
    return normal


def _spo2_drop_delay(t: PatientTrajectory) -> int:
    """Count minutes before SpO2 drops below 92."""
    for i, v in enumerate(t.vitals["spo2"]):
        if v < 92:
            return i
    return len(t.vitals["spo2"])


# ═══════════════════════════════════════════════════
# Helper constants
# ═══════════════════════════════════════════════════

_VITAL_BASE_STD = {
    "hr": 8.0,
    "bp_sys": 10.0,
    "bp_dia": 6.0,
    "rr": 2.5,
    "spo2": 1.2,
    "temp": 0.25,
}


def _clamp_vital(name: str, arr: np.ndarray) -> np.ndarray:
    """Clamp vital values to physiological ranges."""
    ranges = {
        "hr": (30, 200),
        "bp_sys": (40, 250),
        "bp_dia": (20, 180),
        "rr": (4, 50),
        "spo2": (50, 100),
        "temp": (34, 42),
    }
    lo, hi = ranges.get(name, (0, 1e6))
    return np.clip(arr, lo, hi)


def save_hero_cases(heroes: Dict[str, PatientTrajectory], output_dir: str):
    """Save hero cases as CSVs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for hero_name, traj in heroes.items():
        rows = []
        for minute in range(traj.duration_minutes):
            rows.append({
                "minute": minute,
                "patient_id": f"HERO-{hero_name}",
                "hr": round(traj.vitals["hr"][minute], 1),
                "bp_sys": round(traj.vitals["bp_sys"][minute], 1),
                "bp_dia": round(traj.vitals["bp_dia"][minute], 1),
                "rr": round(traj.vitals["rr"][minute], 1),
                "spo2": round(traj.vitals["spo2"][minute], 1),
                "temp": round(traj.vitals["temp"][minute], 2),
            })
        df = pd.DataFrame(rows)
        csv_path = output_path / f"{hero_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[Hero] Saved {hero_name} → {csv_path}")
