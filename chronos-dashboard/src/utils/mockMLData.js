export const mockPatientWithML = {
  patient_id: 'P003',
  vitals: { hr: 84, bp_sys: 112, bp_dia: 72, rr: 17, spo2: 96, temp: 37.2 },
  entropy: {
    ces_adjusted: 0.38,
    ces_raw: 0.32,
    ces_slope_6h: -0.002,
    warmup_complete: true,
    sampen_hr: 0.21,
    sampen_rr: 0.18,
    sampen_bp_sys: 0.32,
    sampen_spo2: 0.45
  },
  drug_context: {
    drug_masking: true,
    active_drugs: [{ drug_name: "Norepinephrine", dose: 0.08, unit: "mcg/kg/min" }]
  },
  ml_predictions: {
    deterioration_risk: {
      risk_1h: 0.12,
      risk_4h: 0.58,
      risk_8h: 0.81,
      model_confidence: "high",
      top_drivers: [
        { feature: "ces_slope_6h", description: "Declining entropy trend", importance: 0.31, direction: "increases_risk" },
        { feature: "sampen_rr", description: "Low respiratory complexity", importance: 0.22, direction: "increases_risk" },
        { feature: "shock_index", description: "Elevated shock index", importance: 0.15, direction: "increases_risk" },
        { feature: "vasopressor_active", description: "Vasopressor dependency", importance: 0.12, direction: "increases_risk" },
        { feature: "spo2_std_6h", description: "Unstable oxygen saturation", importance: 0.08, direction: "increases_risk" }
      ]
    },
    syndrome: {
      primary_syndrome: "Hemodynamic Instability",
      primary_confidence: 0.62,
      secondary_syndrome: "Sepsis-like",
      secondary_confidence: 0.24,
      all_probabilities: {
        sepsis_like: 0.24,
        respiratory_failure: 0.05,
        hemodynamic_instability: 0.62,
        cardiac_instability: 0.06,
        stable: 0.03
      },
      inconclusive: false,
      disclaimer: "Pattern similarity assessment, not a clinical diagnosis"
    },
    warmup_mode: false
  },
  fusion: {
    final_risk_score: 72,
    final_severity: "CRITICAL",
    time_to_event_estimate: "~4 hours",
    component_risks: {
      entropy: 0.62,
      trend: 0.45,
      ml: 0.58,
      masking: 0.30,
      news2: 0.20
    },
    ml_available: true,
    override_applied: null,
    disagreement: null
  },
  detectors: [
    { detector_name: "entropy_threshold", active: true, severity: "WARNING", message: "CES below WARNING threshold (0.38)", contributing_factors: ["ces_adjusted: 0.38"], recommended_action: "Monitor closely" },
    { detector_name: "silent_decline", active: true, severity: "WARNING", message: "Silent decline — entropy falling while vitals appear normal", contributing_factors: ["ces_slope_6h: -0.002", "all_vitals_in_range: true"], recommended_action: "Review patient at bedside" },
    { detector_name: "drug_masking", active: true, severity: "WARNING", message: "Norepinephrine may be masking hemodynamic deterioration", contributing_factors: ["drug: Norepinephrine", "bp_stable: true", "bp_entropy_declining: true"], recommended_action: "Evaluate vasopressor dependency" },
    { detector_name: "respiratory_risk", active: false, severity: "NONE", message: "", contributing_factors: [], recommended_action: "" },
    { detector_name: "hemodynamic", active: true, severity: "WATCH", message: "Hemodynamic instability pattern detected", contributing_factors: ["sampen_bp_sys: 0.32", "shock_index: 0.75"], recommended_action: "Review hemodynamic parameters" },
    { detector_name: "alarm_suppression", active: false, severity: "NONE", message: "", contributing_factors: [], recommended_action: "" },
    { detector_name: "recovery", active: false, severity: "NONE", message: "", contributing_factors: [], recommended_action: "" },
    { detector_name: "data_quality", active: false, severity: "NONE", message: "", contributing_factors: [], recommended_action: "" }
  ],
  recommendations: {
    interventions: [
      { rank: 1, action: "Vasopressor dose adjustment", historical_success_rate: 0.78, similar_cases_count: 142, median_response_time_hours: 1.2, evidence_source: "MIMIC-IV cohort (n=142)" },
      { rank: 2, action: "500mL crystalloid fluid bolus", historical_success_rate: 0.65, similar_cases_count: 203, median_response_time_hours: 0.5, evidence_source: "MIMIC-IV cohort (n=203)" },
      { rank: 3, action: "Prepare for intubation", historical_success_rate: 0.61, similar_cases_count: 87, median_response_time_hours: 0.3, evidence_source: "MIMIC-IV cohort (n=87)" }
    ],
    suggested_tests: [
      { test: "Echocardiogram", reason: "Assess cardiac output" },
      { test: "Serum Lactate", reason: "Tissue perfusion marker" },
      { test: "CVP Assessment", reason: "Volume status evaluation" }
    ]
  }
};

export const mockPatientWarmup = {
  patient_id: 'P005',
  ml_predictions: {
    deterioration_risk: { risk_1h: 0.18, risk_4h: 0.22, risk_8h: 0.30, model_confidence: 'moderate', top_drivers: [] },
    syndrome: null,
    warmup_mode: true
  },
  fusion: {
    final_risk_score: 28,
    final_severity: 'WATCH',
    time_to_event_estimate: 'Unknown (calibrating)',
    ml_available: true,
  },
  detectors: [],
  recommendations: { interventions: [], suggested_tests: [] }
};

export const mockPatientNoML = {
  patient_id: 'P007',
  ml_predictions: { deterioration_risk: null, syndrome: null, warmup_mode: true },
  fusion: { final_risk_score: 42, final_severity: 'WATCH', ml_available: false },
  detectors: [],
  recommendations: { interventions: [], suggested_tests: [] }
};
