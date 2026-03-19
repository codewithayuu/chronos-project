"""
Configuration loader for Project Chronos.
Reads config.yml and validates using Pydantic models.
"""

from pydantic import BaseModel
from typing import List
from pathlib import Path
import yaml


class EntropyWeights(BaseModel):
    heart_rate: float = 0.25
    spo2: float = 0.15
    bp_systolic: float = 0.20
    bp_diastolic: float = 0.10
    resp_rate: float = 0.20
    temperature: float = 0.10


class EntropyThresholds(BaseModel):
    none: float = 0.60
    watch: float = 0.40
    warning: float = 0.20
    critical: float = 0.00


class TrendConfig(BaseModel):
    slope_window: int = 360
    rising_threshold: float = 0.001
    falling_threshold: float = -0.001


class EntropyEngineConfig(BaseModel):
    sampen_m: int = 2
    sampen_r_fraction: float = 0.2
    window_size: int = 300
    min_valid_fraction: float = 0.8
    warmup_points: int = 300
    mse_scales: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    weights: EntropyWeights = EntropyWeights()
    thresholds: EntropyThresholds = EntropyThresholds()
    trend: TrendConfig = TrendConfig()


class DrugFilterConfig(BaseModel):
    weight_reduction_factor: float = 0.5
    tolerance_fraction: float = 0.30
    drug_database_path: str = "data/drug_database.json"


class EvidenceEngineConfig(BaseModel):
    k_neighbors: int = 50
    min_cases_for_recommendation: int = 5
    max_interventions_returned: int = 5
    min_distance_threshold: float = 15.0
    num_synthetic_cases: int = 500
    random_seed: int = 42


class DataReplayConfig(BaseModel):
    speed_multiplier: float = 60.0
    loop: bool = True
    num_filler_patients: int = 5


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]


class AppConfig(BaseModel):
    entropy_engine: EntropyEngineConfig = EntropyEngineConfig()
    drug_filter: DrugFilterConfig = DrugFilterConfig()
    evidence_engine: EvidenceEngineConfig = EvidenceEngineConfig()
    data_replay: DataReplayConfig = DataReplayConfig()
    api: ApiConfig = ApiConfig()


def load_config(path: str = "config.yml") -> AppConfig:
    """Load configuration from YAML file. Falls back to defaults if file not found."""
    config_path = Path(path)
    if config_path.exists():
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raw = {}
        return AppConfig(**raw)
    else:
        print(f"[Config] {path} not found, using defaults.")
        return AppConfig()
