

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class AlertSeverity(str, Enum):
    NONE = "NONE"
    WATCH = "WATCH"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class TrendDirection(str, Enum):
    RISING = "rising"
    STABLE = "stable"
    FALLING = "falling"


# ──────────────────────────────────────────────
# Input Models
# ──────────────────────────────────────────────

class VitalSignRecord(BaseModel):
    """A single vital-sign reading from a bedside monitor (or replay service)."""
    patient_id: str
    timestamp: datetime
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    bp_systolic: Optional[float] = None
    bp_diastolic: Optional[float] = None
    resp_rate: Optional[float] = None
    temperature: Optional[float] = None


# ──────────────────────────────────────────────
# Internal / Output Models
# ──────────────────────────────────────────────

class VitalDetail(BaseModel):
    """Detailed state of a single vital sign including entropy metrics."""
    value: Optional[float] = None
    sampen: Optional[float] = None
    sampen_normalized: Optional[float] = None
    mse: Optional[List[Optional[float]]] = None
    trend: TrendDirection = TrendDirection.STABLE


class VitalsState(BaseModel):
    """Entropy-enriched state of all six vital signs."""
    heart_rate: VitalDetail = Field(default_factory=VitalDetail)
    spo2: VitalDetail = Field(default_factory=VitalDetail)
    bp_systolic: VitalDetail = Field(default_factory=VitalDetail)
    bp_diastolic: VitalDetail = Field(default_factory=VitalDetail)
    resp_rate: VitalDetail = Field(default_factory=VitalDetail)
    temperature: VitalDetail = Field(default_factory=VitalDetail)


class AlertState(BaseModel):
    """Current alert status for a patient."""
    active: bool = False
    severity: AlertSeverity = AlertSeverity.NONE
    message: str = ""
    hours_to_predicted_event: Optional[float] = None
    contributing_vitals: List[str] = Field(default_factory=list)
    drug_masked: bool = False


class DrugEffect(BaseModel):
    """A drug currently active on a patient."""
    drug_name: str
    drug_class: Optional[str] = None
    dose: Optional[float] = None
    unit: Optional[str] = None
    start_time: Optional[datetime] = None
    expected_effects: Optional[Dict[str, Any]] = None


class Intervention(BaseModel):
    """A ranked intervention suggestion from the Evidence Engine."""
    rank: int
    action: str
    historical_success_rate: float
    similar_cases_count: int
    median_response_time_hours: Optional[float] = None
    evidence_source: str = "MIMIC-IV cohort analysis"


class PatientState(BaseModel):
    """
    Complete state of a patient at a point in time.
    This is the primary object that flows from backend to frontend.
    """
    patient_id: str
    timestamp: datetime
    vitals: VitalsState = Field(default_factory=VitalsState)
    composite_entropy: float = 1.0
    composite_entropy_raw: float = 1.0
    active_drugs: List[DrugEffect] = Field(default_factory=list)
    alert: AlertState = Field(default_factory=AlertState)
    interventions: List[Intervention] = Field(default_factory=list)
    calibrating: bool = True
    window_fill: float = 0.0
    
    # Traditional alarm comparison (Phase 1)
    traditional_alarm: bool = False
    traditional_alarm_vitals: List[str] = Field(default_factory=list)
    
    # Clinical scores (Phase 2)
    clinical_scores: Optional[Dict[str, Any]] = None

    # Cross-vital correlations (Phase 4)
    cross_correlations: Optional[Dict[str, Any]] = None
    decoupling_alerts: List[str] = Field(default_factory=list)


class PatientSummary(BaseModel):
    """Lightweight summary for the multi-patient ward view."""
    patient_id: str
    composite_entropy: float = 1.0
    alert_severity: AlertSeverity = AlertSeverity.NONE
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    bp_systolic: Optional[float] = None
    resp_rate: Optional[float] = None
    calibrating: bool = True
    last_update: Optional[datetime] = None


# ──────────────────────────────────────────────
# Drug Database Entry (from drug_database.json)
# ──────────────────────────────────────────────

class DrugDatabaseEntry(BaseModel):
    """A drug profile from the pharmacological lookup table."""
    drug_id: str
    drug_name: str
    drug_class: str
    expected_hr_effect: str          # "decrease" | "increase" | "none"
    expected_hr_magnitude: float     # expected bpm change (signed)
    expected_bp_effect: str
    expected_bp_magnitude: float     # expected mmHg change (signed, applies to systolic)
    expected_rr_effect: str
    expected_rr_magnitude: float
    expected_spo2_effect: str
    expected_spo2_magnitude: float
    onset_minutes: int
    duration_minutes: int
    entropy_impact: str              # "reduces" | "increases" | "none"


# ──────────────────────────────────────────────
# NEW ML Models (added for ML augmentation sidecar)
# ──────────────────────────────────────────────

class MLPredictions(BaseModel):
    """ML prediction outputs for a patient."""
    deterioration_risk: Optional[Dict[str, Any]] = None
    syndrome: Optional[Dict[str, Any]] = None
    warmup_mode: bool = True


class FusionResult(BaseModel):
    """Decision fusion output combining entropy + ML + clinical scores."""
    final_risk_score: int = 0
    final_severity: str = "NONE"
    time_to_event_estimate: str = "Unknown"
    component_risks: Dict[str, Any] = {}
    ml_available: bool = False
    override_applied: Optional[str] = None
    disagreement: Optional[Dict[str, Any]] = None


class DetectorResult(BaseModel):
    """Output from a single detector in the DetectorBank."""
    detector_name: str
    active: bool = False
    severity: str = "NONE"
    message: str = ""
    contributing_factors: List[str] = Field(default_factory=list)
    recommended_action: str = ""


class Recommendations(BaseModel):
    """Evidence-based intervention recommendations and suggested tests."""
    interventions: List[Dict[str, Any]] = Field(default_factory=list)
    suggested_tests: List[Dict[str, Any]] = Field(default_factory=list)

