
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from ..models import VitalSignRecord, DrugEffect
from ..config import AppConfig


class VitalIngestionResponse(BaseModel):
    status: str = "accepted"
    patient_id: str
    window_size: int


class DrugAdminRequest(BaseModel):
    drug_name: str
    drug_class: Optional[str] = None
    dose: Optional[float] = None
    unit: Optional[str] = None
    start_time: Optional[datetime] = None


class AlertAckRequest(BaseModel):
    acknowledged_by: str = "clinician"


def _interpret_scores_vs_entropy(state) -> str:
    """Generate interpretation comparing clinical scores to entropy."""
    if state.clinical_scores is None:
        return "Clinical scores not yet computed."

    news2 = state.clinical_scores.get("news2", {})
    news2_score = news2.get("score", 0)
    news2_risk = news2.get("risk_level", "Unknown")
    ces = state.composite_entropy
    severity = state.alert.severity.value if state.alert else "NONE"

    if severity in ("WARNING", "CRITICAL") and news2_risk in ("None", "Low"):
        return (
            f"CRITICAL INSIGHT: NEWS2 score is {news2_score} ({news2_risk} risk) "
            f"- traditional scoring sees NO problem. But Chronos CES is {ces:.2f} "
            f"({severity}) - entropy analysis detects underlying physiological "
            f"deterioration that standard scores MISS. This is exactly the gap "
            f"Chronos fills."
        )
    elif severity == "NONE" and news2_risk in ("None", "Low"):
        return (
            f"Both Chronos (CES={ces:.2f}) and NEWS2 ({news2_score}) "
            f"indicate stable patient."
        )
    elif severity in ("WARNING", "CRITICAL") and news2_risk in ("Medium", "High"):
        return (
            f"Both systems detect deterioration: NEWS2={news2_score} "
            f"({news2_risk}), Chronos CES={ces:.2f} ({severity}). "
            f"Chronos likely detected this earlier."
        )
    else:
        return (
            f"NEWS2: {news2_score} ({news2_risk}). "
            f"Chronos CES: {ces:.2f} ({severity})."
        )


def _generate_drug_warnings(state, drug_req, effects):
    """Generate warnings about potential drug interactions."""
    warnings = []
    active_drugs = state.active_drugs or []
    for ad in active_drugs:
        if ad.drug_name and ad.drug_name.lower() == drug_req.drug_name.lower():
            warnings.append(f"Patient already receiving {ad.drug_name}")
    if state.alert.severity.value == "CRITICAL":
        warnings.append("Patient is in CRITICAL state. Exercise extreme caution.")
    if state.alert.drug_masked:
        warnings.append("Drug masking already detected. Additional drugs may further obscure true condition.")
    for effect in effects:
        pv = effect.get("predicted_value")
        vk = effect.get("vital_key", "")
        if pv is not None:
            if vk == "heart_rate" and (pv < 45 or pv > 140):
                warnings.append(f"Predicted heart rate {pv:.0f} bpm is dangerous")
            if vk == "bp_systolic" and (pv < 80 or pv > 200):
                warnings.append(f"Predicted systolic BP {pv:.0f} mmHg is dangerous")
            if vk == "resp_rate" and (pv < 6 or pv > 35):
                warnings.append(f"Predicted respiratory rate {pv:.0f}/min is dangerous")
            if vk == "spo2" and pv < 88:
                warnings.append(f"Predicted SpO2 {pv:.0f}% is dangerously low")
    return warnings


def create_router() -> APIRouter:
    router = APIRouter()

    def _mgr(request: Request):
        return request.app.state.manager

    # -- Vital Signs --

    @router.post("/vitals", response_model=VitalIngestionResponse)
    def ingest_vital(record: VitalSignRecord, request: Request):
        manager = _mgr(request)
        manager.process_vital(record)
        window = manager.entropy_engine.patients.get(record.patient_id)
        return VitalIngestionResponse(
            patient_id=record.patient_id,
            window_size=window.current_size if window else 0,
        )

    # -- Patients --

    @router.get("/patients")
    def list_patients(request: Request):
        manager = _mgr(request)
        summaries = manager.get_all_summaries()
        return [s.model_dump() for s in summaries]

    @router.get("/patients/{patient_id}")
    def get_patient(patient_id: str, request: Request):
        manager = _mgr(request)
        state = manager.get_patient_state(patient_id)
        if state is None:
            raise HTTPException(
                status_code=404, detail=f"Patient {patient_id} not found"
            )
        return state.model_dump()

    @router.get("/patients/{patient_id}/history")
    def get_patient_history(
        patient_id: str, request: Request, hours: int = 6
    ):
        manager = _mgr(request)
        history = manager.get_patient_history(patient_id, hours)
        if not history:
            raise HTTPException(
                status_code=404,
                detail=f"No history for patient {patient_id}",
            )
        return [s.model_dump() for s in history]

    @router.get("/patients/{patient_id}/drugs")
    def get_patient_drugs(patient_id: str, request: Request):
        manager = _mgr(request)
        drugs = manager.get_patient_drugs(patient_id)
        return [d.model_dump() for d in drugs]

    @router.post("/patients/{patient_id}/drugs")
    def add_drug(
        patient_id: str, drug_req: DrugAdminRequest, request: Request
    ):
        manager = _mgr(request)
        drug = DrugEffect(
            drug_name=drug_req.drug_name,
            drug_class=drug_req.drug_class,
            dose=drug_req.dose,
            unit=drug_req.unit,
            start_time=drug_req.start_time or datetime.utcnow(),
        )
        manager.add_drug(patient_id, drug)
        return {
            "status": "recorded",
            "patient_id": patient_id,
            "drug": drug_req.drug_name,
        }

    # -- Alerts --

    @router.get("/alerts")
    def get_alerts(request: Request):
        manager = _mgr(request)
        return manager.get_all_alerts()

    @router.post("/alerts/{alert_id}/acknowledge")
    def acknowledge_alert(
        alert_id: str, ack: AlertAckRequest, request: Request
    ):
        manager = _mgr(request)
        found = manager.acknowledge_alert(alert_id, ack.acknowledged_by)
        if not found:
            raise HTTPException(
                status_code=404, detail=f"Alert {alert_id} not found"
            )
        return {"status": "acknowledged", "alert_id": alert_id}

    # -- System --

    @router.get("/system/health")
    def health_check(request: Request):
        manager = _mgr(request)
        health = manager.get_health()
        health["ws_clients"] = request.app.state.ws_manager.client_count
        return health

    # -- Analytics (Phase 1-3) --

    @router.get("/analytics/validation")
    def get_validation_report(request: Request):
        """
        Get validation metrics comparing Chronos vs traditional alarms.
        Results are computed in background at startup.
        Returns cached results or computing status.
        """
        from ..analytics.validator import ValidationEngine
        result = ValidationEngine.get_cached_report()
        return result

    @router.get("/analytics/alarm-fatigue")
    def get_alarm_fatigue(request: Request):
        """
        Get real-time alarm fatigue comparison statistics.
        Shows how many traditional alarms would have fired vs
        how many Chronos alerts were generated.
        """
        manager = _mgr(request)
        return manager.alarm_tracker.get_statistics()

    @router.get("/analytics/clinical-scores/{patient_id}")
    def get_clinical_scores(patient_id: str, request: Request):
        """
        Get current clinical scores (NEWS2, qSOFA) for a patient
        with comparison to Chronos entropy analysis.
        """
        manager = _mgr(request)
        state = manager.get_patient_state(patient_id)

        if state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {patient_id} not found",
            )

        return {
            "patient_id": patient_id,
            "clinical_scores": state.clinical_scores,
            "composite_entropy": state.composite_entropy,
            "alert_severity": (
                state.alert.severity.value if state.alert else "NONE"
            ),
            "interpretation": _interpret_scores_vs_entropy(state),
        }

    @router.get("/analytics/correlations/{patient_id}")
    def get_correlations(patient_id: str, request: Request):
        """
        Get cross-vital correlation analysis for a patient.
        Shows organ system coupling status and decoupling alerts.
        """
        manager = _mgr(request)
        state = manager.get_patient_state(patient_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {patient_id} not found",
            )
        correlations = manager.cross_vital.compute_correlations(
            patient_id
        )
        summary = manager.cross_vital.get_decoupling_summary(
            patient_id
        )
        return {
            "patient_id": patient_id,
            "correlations": correlations,
            "summary": summary,
        }

    @router.get("/patients/{patient_id}/narrative")
    def get_narrative(patient_id: str, request: Request):
        """
        Get a plain-English clinical narrative explaining
        patient's current status, entropy analysis,
        drug context, organ coupling, and recommendations.
        """
        from ..analytics.narrative import NarrativeGenerator

        manager = _mgr(request)
        state = manager.get_patient_state(patient_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {patient_id} not found",
            )
        history = manager.get_patient_history(patient_id)
        decoupling = manager.cross_vital.get_decoupling_summary(
            patient_id
        )

        gen = NarrativeGenerator()
        narrative = gen.generate(
            state,
            history_length_minutes=len(history),
            decoupling_summary=decoupling,
        )
        return narrative

    # -- Phase 6: Digital Twin, AI, Voice, Charts --

    @router.get("/digital-twin/{patient_id}")
    def get_digital_twin(patient_id: str, request: Request):
        """
        Get body-region-mapped data for 3D digital twin model.
        Maps entropy, vitals, and correlations to body regions
        with danger levels and colors.
        """
        from ..analytics.digital_twin import DigitalTwinMapper

        manager = _mgr(request)
        state = manager.get_patient_state(patient_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {patient_id} not found",
            )
        correlations = manager.cross_vital.compute_correlations(
            patient_id
        )
        decoupling = manager.cross_vital.get_decoupling_summary(
            patient_id
        )

        mapper = DigitalTwinMapper()
        return mapper.map_patient(
            state, correlations, decoupling
        )

    @router.post("/analytics/ai-analysis/{patient_id}")
    def get_ai_analysis(patient_id: str, request: Request):
        """
        Get AI-enhanced clinical analysis for a patient.
        Uses Gemini or Grok API if configured, falls back to
        template-based narrative.

        Set GEMINI_API_KEY or GROK_API_KEY environment variable
        to enable AI analysis.
        """
        from ..analytics.ai_analysis import analyze_patient

        manager = _mgr(request)
        state = manager.get_patient_state(patient_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {patient_id} not found",
            )

        decoupling = manager.cross_vital.get_decoupling_summary(
            patient_id
        )
        extra = {"decoupling": decoupling}

        return analyze_patient(state, extra_context=extra)

    @router.get("/voice-alert/{patient_id}")
    def get_voice_alert(patient_id: str, request: Request):
        """
        Get voice-ready text for TTS synthesis (Sarvam AI).
        Returns formatted text with priority, rate, and pitch
        settings for natural-sounding clinical alerts.
        """
        from ..analytics.voice_formatter import VoiceFormatter
        from ..analytics.narrative import NarrativeGenerator

        manager = _mgr(request)
        state = manager.get_patient_state(patient_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {patient_id} not found",
            )

        gen = NarrativeGenerator()
        decoupling = manager.cross_vital.get_decoupling_summary(
            patient_id
        )
        narrative = gen.generate(
            state, decoupling_summary=decoupling
        )

        formatter = VoiceFormatter()
        return formatter.format_alert(
            state, narrative.get("full_text")
        )

    @router.get("/analytics/charts/{patient_id}")
    def get_patient_charts(patient_id: str, request: Request):
        """
        Get pre-formatted chart data for a patient.
        Returns arrays ready for Recharts/Chart.js rendering.
        """
        from ..analytics.chart_data import ChartDataFormatter

        manager = _mgr(request)
        state = manager.get_patient_state(patient_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {patient_id} not found",
            )

        history = manager.get_patient_history(patient_id)
        correlations = manager.cross_vital.compute_correlations(
            patient_id
        )

        formatter = ChartDataFormatter()
        return formatter.patient_charts(
            state, history, correlations
        )

    @router.get("/analytics/dashboard-summary")
    def get_dashboard_summary(request: Request):
        """
        Get system-wide dashboard data including:
        - Patient severity distribution
        - Alarm comparison (traditional vs Chronos)
        - Patient entropy ranking
        - Validation metrics
        """
        from ..analytics.chart_data import ChartDataFormatter
        from ..analytics.validator import ValidationEngine

        manager = _mgr(request)
        formatter = ChartDataFormatter()

        all_states = manager.latest_states
        alarm_stats = manager.alarm_tracker.get_statistics()

        validation = ValidationEngine.get_cached_report()
        validation_data = None
        if (
            validation
            and validation.get("status") == "complete"
        ):
            validation_data = validation.get("report")

        charts = formatter.system_dashboard(
            all_states, alarm_stats, validation_data
        )

        charts["system_health"] = manager.get_health()
        charts["alarm_fatigue"] = alarm_stats

        return charts

    # -- Phase B: Drug Simulation Lab --

    @router.get("/drugs/list")
    def list_drugs(request: Request):
        """List all drugs in the database with their effects."""
        manager = _mgr(request)
        db = manager.drug_db
        drugs = []
        try:
            # Try different access patterns based on DrugDatabase implementation
            if hasattr(db, 'drugs') and isinstance(db.drugs, list):
                for entry in db.drugs:
                    drugs.append({
                        "drug_name": entry.drug_name,
                        "drug_class": entry.drug_class,
                        "expected_hr_effect": getattr(entry, 'expected_hr_effect', ''),
                        "expected_hr_magnitude": getattr(entry, 'expected_hr_magnitude', 0),
                        "expected_bp_effect": getattr(entry, 'expected_bp_effect', ''),
                        "expected_bp_magnitude": getattr(entry, 'expected_bp_magnitude', 0),
                        "expected_rr_effect": getattr(entry, 'expected_rr_effect', ''),
                        "expected_rr_magnitude": getattr(entry, 'expected_rr_magnitude', 0),
                        "entropy_impact": getattr(entry, 'entropy_impact', ''),
                        "onset_minutes": getattr(entry, 'onset_minutes', 0),
                        "duration_minutes": getattr(entry, 'duration_minutes', 0),
                    })
            elif hasattr(db, '_by_name'):
                for name, entry in db._by_name.items():
                    drugs.append({
                        "drug_name": name,
                        "drug_class": entry.drug_class,
                        "expected_hr_effect": getattr(entry, 'expected_hr_effect', ''),
                        "expected_hr_magnitude": getattr(entry, 'expected_hr_magnitude', 0),
                        "expected_bp_effect": getattr(entry, 'expected_bp_effect', ''),
                        "expected_bp_magnitude": getattr(entry, 'expected_bp_magnitude', 0),
                        "expected_rr_effect": getattr(entry, 'expected_rr_effect', ''),
                        "expected_rr_magnitude": getattr(entry, 'expected_rr_magnitude', 0),
                        "entropy_impact": getattr(entry, 'entropy_impact', ''),
                        "onset_minutes": getattr(entry, 'onset_minutes', 0),
                        "duration_minutes": getattr(entry, 'duration_minutes', 0),
                    })
        except Exception as e:
            return {"error": str(e), "drugs": []}
        
        return {"drugs": drugs, "count": len(drugs)}

    @router.get("/drugs/search")
    def search_drugs(q: str, request: Request):
        """Search drugs by name (partial match)."""
        manager = _mgr(request)
        all_drugs = list_drugs(request)
        if "error" in all_drugs:
            return all_drugs
        
        query = q.lower().strip()
        if not query:
            return all_drugs
        
        filtered = [
            d for d in all_drugs["drugs"]
            if query in d.get("drug_name", "").lower()
            or query in d.get("drug_class", "").lower()
        ]
        return {"drugs": filtered, "count": len(filtered), "query": q}

    @router.post("/drugs/simulate/{patient_id}")
    def simulate_drug(patient_id: str, drug_req: DrugAdminRequest, request: Request):
        """
        Simulate drug effects WITHOUT actually administering.
        Returns predicted vital sign changes and entropy impact
        based on the drug's known effects and the patient's current state.
        """
        manager = _mgr(request)
        state = manager.get_patient_state(patient_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        # Get drug info from database
        drug_info = None
        db = manager.drug_db
        try:
            drug_info = db.lookup(drug_req.drug_name)
        except Exception:
            pass
        
        if drug_info is None:
            return {
                "patient_id": patient_id,
                "drug_name": drug_req.drug_name,
                "found_in_database": False,
                "message": f"Drug '{drug_req.drug_name}' not found in database. Effects cannot be predicted.",
                "predicted_effects": [],
            }
        
        # Extract current vitals
        current_vitals = {}
        for vname in ["heart_rate", "spo2", "bp_systolic", "bp_diastolic", "resp_rate", "temperature"]:
            detail = getattr(state.vitals, vname, None)
            if detail and detail.value is not None:
                current_vitals[vname] = {
                    "current_value": detail.value,
                    "current_entropy": detail.sampen_normalized,
                }
        
        # Predict effects
        effects = []
        effect_fields = [
            ("heart_rate", "expected_hr_effect", "expected_hr_magnitude", "bpm"),
            ("bp_systolic", "expected_bp_effect", "expected_bp_magnitude", "mmHg"),
            ("resp_rate", "expected_rr_effect", "expected_rr_magnitude", "/min"),
            ("spo2", "expected_spo2_effect", "expected_spo2_magnitude", "%"),
        ]
        
        dose_factor = 1.0
        if drug_req.dose is not None:
            dose_factor = min(2.0, max(0.5, drug_req.dose / 10.0))
        
        for vital_name, effect_field, mag_field, unit in effect_fields:
            effect_dir = getattr(drug_info, effect_field, "none")
            magnitude = getattr(drug_info, mag_field, 0) or 0
            
            if effect_dir == "none" or magnitude == 0:
                continue
            
            current = current_vitals.get(vital_name, {})
            current_val = current.get("current_value")
            current_ent = current.get("current_entropy")
            
            predicted_change = magnitude * dose_factor
            predicted_value = (current_val + predicted_change) if current_val else None
            
            entropy_impact = getattr(drug_info, "entropy_impact", "none")
            entropy_change = -0.15 if entropy_impact == "reduces" else (0.05 if entropy_impact == "increases" else 0)
            predicted_entropy = max(0, min(1, (current_ent or 0.5) + entropy_change)) if current_ent is not None else None
            
            effects.append({
                "vital_sign": vital_name.replace("_", " ").title(),
                "vital_key": vital_name,
                "direction": effect_dir,
                "current_value": round(current_val, 1) if current_val else None,
                "predicted_change": round(predicted_change, 1),
                "predicted_value": round(predicted_value, 1) if predicted_value else None,
                "unit": unit,
                "current_entropy": round(current_ent, 3) if current_ent else None,
                "predicted_entropy": round(predicted_entropy, 3) if predicted_entropy else None,
                "entropy_impact": entropy_impact,
            })
        
        # Historical effectiveness from evidence engine
        effectiveness = None
        try:
            from ..analytics.clinical_scores import ClinicalScores
            interventions = state.interventions or []
            matching = [i for i in interventions if drug_req.drug_name.lower() in i.action.lower()]
            if matching:
                effectiveness = {
                    "historical_success_rate": matching[0].historical_success_rate,
                    "similar_cases": matching[0].similar_cases_count,
                    "source": "Evidence Engine (KNN historical matching)",
                }
        except Exception:
            pass
        
        onset = getattr(drug_info, "onset_minutes", 0) or 0
        duration = getattr(drug_info, "duration_minutes", 0) or 0
        
        return {
            "patient_id": patient_id,
            "drug_name": drug_req.drug_name,
            "drug_class": drug_info.drug_class,
            "dose": drug_req.dose,
            "unit": drug_req.unit,
            "found_in_database": True,
            "onset_minutes": onset,
            "duration_minutes": duration,
            "current_patient_ces": round(state.composite_entropy, 3),
            "current_severity": state.alert.severity.value,
            "predicted_effects": effects,
            "entropy_impact": getattr(drug_info, "entropy_impact", "none"),
            "historical_effectiveness": effectiveness,
            "warnings": _generate_drug_warnings(state, drug_req, effects),
        }

    return router
