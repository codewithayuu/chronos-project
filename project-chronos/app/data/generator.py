"""
Synthetic ICU Data Generator.

Creates hero cases (patients with specific deterioration patterns)
and filler patients (stable patients for ward view).

Each hero case demonstrates a key Chronos capability:
  Hero 1: "Silent Sepsis" — entropy drops 4+ hours before vitals crash
  Hero 2: "Masked Respiratory Failure" — Propofol hides RR decline
  Hero 3: "Controlled Cardiac Crash" — Metoprolol masks HR entropy loss
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

from ..models import VitalSignRecord, DrugEffect

# ──────────────────────────────────────────────


# Data structures


# ──────────────────────────────────────────────

@dataclass
class DrugEvent:
    """A drug administration event at a specific time offset."""
    minute: int
    drug_name: str
    drug_class: str
    dose: float
    unit: str

@dataclass
class PhasePoint:
    """
    A keypoint in a patient's trajectory.
    The generator interpolates between consecutive keypoints.
    """
    minute: int
    variability: float       # 1.0 = healthy complexity, 0.05 = near-flat
    hr_mean: float
    spo2_mean: float
    bp_sys_mean: float
    bp_dia_mean: float
    rr_mean: float
    temp_mean: float

@dataclass
class PatientCase:
    """A complete patient case with vital records and drug events."""
    patient_id: str
    name: str
    description: str
    records: List[VitalSignRecord] = field(default_factory=list)
    drug_events: List[DrugEvent] = field(default_factory=list)
    duration_minutes: int = 720

# ──────────────────────────────────────────────


# Interpolation helpers


# ──────────────────────────────────────────────

def _smooth_interp(t: float, t0: float, t1: float, v0: float, v1: float) -> float:
    """Smooth cosine interpolation between v0 and v1 over [t0, t1]."""
    if t <= t0:
        return v0
    if t >= t1:
        return v1
    frac = (t - t0) / (t1 - t0)
    frac = 0.5 * (1.0 - np.cos(frac * np.pi))
    return v0 + (v1 - v0) * frac

def _get_params_at_minute(minute: int, keypoints: List[PhasePoint]) -> Dict[str, float]:
    """Interpolate all parameters at a given minute from keypoints."""
    if minute <= keypoints[0].minute:
        kp = keypoints[0]
        return {
            "variability": kp.variability, "hr": kp.hr_mean, "spo2": kp.spo2_mean,
            "bp_sys": kp.bp_sys_mean, "bp_dia": kp.bp_dia_mean,
            "rr": kp.rr_mean, "temp": kp.temp_mean,
        }

    if minute >= keypoints[-1].minute:
        kp = keypoints[-1]
        return {
            "variability": kp.variability, "hr": kp.hr_mean, "spo2": kp.spo2_mean,
            "bp_sys": kp.bp_sys_mean, "bp_dia": kp.bp_dia_mean,
            "rr": kp.rr_mean, "temp": kp.temp_mean,
        }

    # Find surrounding keypoints
    for i in range(len(keypoints) - 1):
        if keypoints[i].minute <= minute < keypoints[i + 1].minute:
            kp0, kp1 = keypoints[i], keypoints[i + 1]
            t0, t1 = float(kp0.minute), float(kp1.minute)
            t = float(minute)
            return {
                "variability": _smooth_interp(t, t0, t1, kp0.variability, kp1.variability),
                "hr": _smooth_interp(t, t0, t1, kp0.hr_mean, kp1.hr_mean),
                "spo2": _smooth_interp(t, t0, t1, kp0.spo2_mean, kp1.spo2_mean),
                "bp_sys": _smooth_interp(t, t0, t1, kp0.bp_sys_mean, kp1.bp_sys_mean),
                "bp_dia": _smooth_interp(t, t0, t1, kp0.bp_dia_mean, kp1.bp_dia_mean),
                "rr": _smooth_interp(t, t0, t1, kp0.rr_mean, kp1.rr_mean),
                "temp": _smooth_interp(t, t0, t1, kp0.temp_mean, kp1.temp_mean),
            }

    kp = keypoints[-1]
    return {
        "variability": kp.variability, "hr": kp.hr_mean, "spo2": kp.spo2_mean,
        "bp_sys": kp.bp_sys_mean, "bp_dia": kp.bp_dia_mean,
        "rr": kp.rr_mean, "temp": kp.temp_mean,
    }

def _generate_vital(
    minute: int, rng: np.random.RandomState,
    mean: float, base_std: float, variability: float,
    phase_offset: float = 0.0,
    clamp_min: float = -999, clamp_max: float = 999,
) -> float:
    """
    Generate a single vital sign value.

    At high variability (healthy): dominated by random noise → high entropy.
    At low variability (sick): dominated by periodic component → low entropy.
    """
    noise = rng.randn() * base_std * variability
    periodic = np.sin((minute + phase_offset) * 0.1) * base_std * 0.3 * (1.0 - variability)
    circadian = np.sin(minute / 720.0 * np.pi) * base_std * 0.05

    value = mean + noise + periodic + circadian
    return float(np.clip(value, clamp_min, clamp_max))

# ──────────────────────────────────────────────


# Standard deviations for each vital (base noise level)


# ──────────────────────────────────────────────

VITAL_BASE_STD = {
    "hr": 8.0,
    "spo2": 1.2,
    "bp_sys": 10.0,
    "bp_dia": 6.0,
    "rr": 2.5,
    "temp": 0.25,
}

# ──────────────────────────────────────────────


# Hero Case Definitions


# ──────────────────────────────────────────────

class DataGenerator:
    """Generates synthetic ICU patient data for Project Chronos demo."""

    @staticmethod
    def hero_case_1(base_time: datetime, rng: np.random.RandomState) -> PatientCase:
        """
        HERO CASE 1: "The Silent Decline" — Sepsis

        Story: Patient admitted post-surgery, initially stable.
        Entropy drops 4+ hours before vital signs crash.
        Traditional monitors show green; Chronos shows orange/red.

        Timeline (720 minutes = 12 hours):
          0-300:   Stable, healthy variability (warmup + baseline)
          300-330: Smooth transition to reduced complexity
          330-520: ENTROPY DROPS but vitals stay in normal range ← KEY WINDOW
          520-580: Vitals start shifting (HR↑, BP↓, RR↑)
          580-660: Overt deterioration
          660-720: Crisis (tachycardia, hypotension)
        """
        keypoints = [
            PhasePoint(0,   1.0, 78, 97, 120, 80, 15, 37.0),
            PhasePoint(300, 1.0, 78, 97, 120, 80, 15, 37.0),
            PhasePoint(330, 0.15, 80, 97, 118, 78, 16, 37.1),   # entropy drops
            PhasePoint(520, 0.12, 82, 96, 115, 76, 18, 37.3),   # still "normal" values
            PhasePoint(580, 0.08, 98, 94, 100, 65, 24, 37.8),   # values start shifting
            PhasePoint(660, 0.05, 115, 90, 82, 55, 30, 38.2),   # overt crisis
            PhasePoint(720, 0.03, 125, 87, 75, 48, 34, 38.5),   # severe
        ]
        drug_events = [
            DrugEvent(480, "Norepinephrine", "vasopressor", 0.08, "mcg/kg/min"),
        ]
        return DataGenerator._build_case(
            "BED-01", "Silent Sepsis", 
            "Entropy drops 4+ hours before vital signs crash. "
            "Traditional monitors stay green while Chronos detects decline.",
            720, keypoints, drug_events, base_time, rng,
        )

    @staticmethod
    def hero_case_2(base_time: datetime, rng: np.random.RandomState) -> PatientCase:
        """
        HERO CASE 2: "The Masked Patient" — Propofol Masking

        Story: Post-surgical patient on Propofol sedation.
        Propofol keeps HR/RR looking stable while respiratory
        entropy progressively drops. Drug masking demonstration.

        Timeline:
          0-120:   Post-surgical, stable
          120:     Propofol started
          120-350: Propofol keeps vitals looking stable
          350-500: Respiratory entropy very low, values still "OK"
          500-600: SpO2 starts dropping despite stable RR
          600-720: Respiratory failure
        """
        keypoints = [
            PhasePoint(0,   1.0, 82, 98, 125, 82, 16, 36.9),
            PhasePoint(120, 1.0, 82, 98, 125, 82, 16, 36.9),
            PhasePoint(150, 0.6, 74, 97, 115, 76, 13, 36.8),   # propofol effect
            PhasePoint(350, 0.12, 72, 96, 112, 74, 12, 36.7),   # entropy very low
            PhasePoint(500, 0.08, 70, 93, 108, 72, 11, 36.6),   # SpO2 dropping
            PhasePoint(600, 0.05, 68, 88, 100, 68, 9, 36.5),   # respiratory failure
            PhasePoint(720, 0.03, 65, 83, 95, 62, 7, 36.4),   # severe
        ]
        drug_events = [
            DrugEvent(120, "Propofol", "sedative", 50, "mcg/kg/min"),
        ]
        return DataGenerator._build_case(
            "BED-02", "Masked Respiratory Failure",
            "Propofol sedation masks respiratory decline. "
            "Chronos detects entropy drop and flags drug masking.",
            720, keypoints, drug_events, base_time, rng,
        )

    @staticmethod
    def hero_case_3(base_time: datetime, rng: np.random.RandomState) -> PatientCase:
        """
        HERO CASE 3: "The Controlled Crash" — Beta-Blocker Masking

        Story: Patient on Metoprolol for tachycardia.
        HR stays controlled at 70 due to beta-blocker, but HR
        entropy drops. Eventually cardiac decompensation.

        Timeline:
          0-180:   Tachycardic but controlled with Metoprolol
          180:     Metoprolol dose increased
          180-420: HR value stable (~70) but entropy dropping
          420-560: BP starts falling despite controlled HR
          560-660: Cardiac decompensation
          660-720: Critical
        """
        keypoints = [
            PhasePoint(0,   0.9, 88, 96, 128, 85, 17, 37.1),
            PhasePoint(180, 0.9, 88, 96, 128, 85, 17, 37.1),
            PhasePoint(210, 0.5, 72, 96, 122, 80, 16, 37.0),   # metoprolol kicks in
            PhasePoint(420, 0.10, 70, 95, 118, 78, 16, 37.0),   # HR stable, entropy low
            PhasePoint(560, 0.07, 68, 93, 95, 62, 20, 37.2),   # BP falling
            PhasePoint(660, 0.04, 55, 90, 80, 52, 24, 37.4),   # bradycardia, hypotension
            PhasePoint(720, 0.03, 48, 86, 70, 45, 28, 37.5),   # critical
        ]
        drug_events = [
            DrugEvent(30, "Metoprolol", "beta_blocker", 5.0, "mg"),
            DrugEvent(180, "Metoprolol", "beta_blocker", 10.0, "mg"),
            DrugEvent(560, "Dobutamine", "inotrope", 5.0, "mcg/kg/min"),
        ]
        return DataGenerator._build_case(
            "BED-03", "Controlled Cardiac Crash",
            "Metoprolol masks HR entropy loss. HR looks controlled "
            "but cardiac function deteriorates underneath.",
            720, keypoints, drug_events, base_time, rng,
        )

    @staticmethod
    def stable_patient(
        patient_id: str,
        base_time: datetime,
        rng: np.random.RandomState,
        hr_base: float = 75,
        bp_base: float = 120,
        variability: float = 0.95,
    ) -> PatientCase:
        """Generate a stable patient for ward view filler."""
        keypoints = [
            PhasePoint(0,   variability, hr_base, 97, bp_base, bp_base * 0.67, 15, 37.0),
            PhasePoint(720, variability, hr_base, 97, bp_base, bp_base * 0.67, 15, 37.0),
        ]
        return DataGenerator._build_case(
            patient_id, f"Stable Patient {patient_id}",
            "Stable patient with normal physiological complexity.",
            720, keypoints, [], base_time, rng,
        )

    @staticmethod
    def generate_demo_dataset(
        base_time: Optional[datetime] = None,
        seed: int = 42,
        num_filler: int = 5,
        duration_minutes: int = 720,
    ) -> List[PatientCase]:
        """
        Generate complete demo dataset with hero cases + filler patients.

        Returns list of PatientCase objects ready for replay.
        """
        if base_time is None:
            base_time = datetime.utcnow()

        rng = np.random.RandomState(seed)
        cases = []

        # Hero cases
        cases.append(DataGenerator.hero_case_1(base_time, rng))
        cases.append(DataGenerator.hero_case_2(base_time, rng))
        cases.append(DataGenerator.hero_case_3(base_time, rng))

        # Filler patients with varying profiles
        filler_profiles = [
            {"hr_base": 72, "bp_base": 118, "variability": 1.0},
            {"hr_base": 80, "bp_base": 130, "variability": 0.90},
            {"hr_base": 68, "bp_base": 115, "variability": 0.95},
            {"hr_base": 85, "bp_base": 125, "variability": 0.85},
            {"hr_base": 76, "bp_base": 122, "variability": 0.92},
        ]

        for i in range(min(num_filler, len(filler_profiles))):
            profile = filler_profiles[i]
            cases.append(DataGenerator.stable_patient(
                f"BED-{i + 4:02d}", base_time, rng, **profile,
            ))

        print(f"[DataGenerator] Generated {len(cases)} patient cases "
              f"({3} hero + {len(cases) - 3} filler)")
        return cases

    # ──────────────────────────────────────────────
    # Internal builder
    # ──────────────────────────────────────────────

    @staticmethod
    def _build_case(
        patient_id: str,
        name: str,
        description: str,
        duration: int,
        keypoints: List[PhasePoint],
        drug_events: List[DrugEvent],
        base_time: datetime,
        rng: np.random.RandomState,
    ) -> PatientCase:
        """Build a complete PatientCase from keypoints and drug events."""
        records = []
        # Each vital gets a unique phase offset so signals aren't correlated
        offsets = {"hr": 0, "spo2": 50, "bp_sys": 100, "bp_dia": 130, "rr": 200, "temp": 300}

        for minute in range(duration):
            params = _get_params_at_minute(minute, keypoints)
            v = params["variability"]
            ts = base_time + timedelta(minutes=minute)

            hr = _generate_vital(minute, rng, params["hr"], VITAL_BASE_STD["hr"],
                                 v, offsets["hr"], 30, 200)
            spo2 = _generate_vital(minute, rng, params["spo2"], VITAL_BASE_STD["spo2"],
                                   v, offsets["spo2"], 50, 100)
            bp_sys = _generate_vital(minute, rng, params["bp_sys"], VITAL_BASE_STD["bp_sys"],
                                     v, offsets["bp_sys"], 40, 250)
            bp_dia = _generate_vital(minute, rng, params["bp_dia"], VITAL_BASE_STD["bp_dia"],
                                     v, offsets["bp_dia"], 20, 180)
            rr = _generate_vital(minute, rng, params["rr"], VITAL_BASE_STD["rr"],
                                 v, offsets["rr"], 4, 50)
            temp = _generate_vital(minute, rng, params["temp"], VITAL_BASE_STD["temp"],
                                   v, offsets["temp"], 34, 42)

            records.append(VitalSignRecord(
                patient_id=patient_id,
                timestamp=ts,
                heart_rate=round(hr, 1),
                spo2=round(spo2, 1),
                bp_systolic=round(bp_sys, 1),
                bp_diastolic=round(bp_dia, 1),
                resp_rate=round(rr, 1),
                temperature=round(temp, 1),
            ))

        case = PatientCase(
            patient_id=patient_id,
            name=name,
            description=description,
            records=records,
            drug_events=drug_events,
            duration_minutes=duration,
        )
        return case
