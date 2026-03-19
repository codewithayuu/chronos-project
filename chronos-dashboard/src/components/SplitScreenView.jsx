import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ArrowLeft,
  Heartbeat,
  Drop,
  Wind,
  Thermometer,
  CheckCircle,
  WarningOctagon,
  Clock,
  Lightning,
  SpeakerHigh,
} from '@phosphor-icons/react';
import CESGauge from './CESGauge';
import EntropyBars from './EntropyBars';
import AnimatedNumber from './AnimatedNumber';
import { API_BASE, SEVERITY_CONFIG } from '../utils/constants';
import { formatVital } from '../utils/helpers';
import {
  pageVariants,
  detailPanelVariants,
} from '../utils/animations';
import './SplitScreenView.css';

function TraditionalVitalRow({ label, icon, value, unit, low, high }) {
  const inRange = value == null || (value >= low && (high == null || value <= high));
  return (
    <div className={`trad-vital-row ${!inRange ? 'trad-vital-alarm' : ''}`}>
      <div className="trad-vital-label">
        {icon}
        <span>{label}</span>
      </div>
      <div className="trad-vital-value-group">
        <span className={`trad-vital-value ${!inRange ? 'trad-value-alarm' : ''}`}>
          {formatVital(value)}
        </span>
        <span className="trad-vital-unit">{unit}</span>
      </div>
      {!inRange && (
        <motion.div
          className="trad-alarm-indicator"
          animate={{ opacity: [1, 0.3, 1] }}
          transition={{ duration: 0.8, repeat: Infinity }}
        >
          <SpeakerHigh size={14} weight="fill" />
          <span>ALARM</span>
        </motion.div>
      )}
      {inRange && (
        <div className="trad-ok-indicator">
          <CheckCircle size={14} weight="fill" />
        </div>
      )}
    </div>
  );
}

function SplitScreenView({ patients }) {
  const { patientId } = useParams();
  const navigate = useNavigate();
  const [narrative, setNarrative] = useState(null);

  // Fetch narrative for Chronos side
  const fetchNarrative = useCallback(async () => {
    if (!patientId) return;
    try {
      const res = await fetch(`${API_BASE}/api/v1/patients/${patientId}/narrative`);
      if (res.ok) setNarrative(await res.json());
    } catch (err) {
      console.error('[API] Narrative error:', err);
    }
  }, [patientId]);

  useEffect(() => {
    fetchNarrative();
    const interval = setInterval(fetchNarrative, 20000);
    return () => clearInterval(interval);
  }, [fetchNarrative]);

  // Count traditional alarms
  const patient = patients[patientId];

  const traditionalAlarmCount = useMemo(() => {
    if (!patient?.vitals) return 0;
    let count = 0;
    const checks = [
      { key: 'heart_rate', low: 50, high: 120 },
      { key: 'spo2', low: 90, high: null },
      { key: 'bp_systolic', low: 90, high: 180 },
      { key: 'resp_rate', low: 8, high: 30 },
      { key: 'temperature', low: 35.5, high: 38.5 },
    ];
    checks.forEach(({ key, low, high }) => {
      const val = patient.vitals[key]?.value;
      if (val != null && (val < low || (high != null && val > high))) count++;
    });
    return count;
  }, [patient]);

  if (!patient) {
    return (
      <motion.div className="split-loading" variants={pageVariants} initial="initial" animate="animate" exit="exit">
        <h2>Loading patient data...</h2>
        <motion.button className="split-back-btn" onClick={() => navigate('/')}>
          <ArrowLeft size={14} weight="bold" /> Back to Ward
        </motion.button>
      </motion.div>
    );
  }

  const severity = patient.alert?.severity || 'NONE';
  const config = SEVERITY_CONFIG[severity] || SEVERITY_CONFIG.NONE;
  const vitals = patient.vitals || {};
  const ces = patient.composite_entropy;
  const alert = patient.alert || {};
  const scores = patient.clinical_scores || {};
  const news2 = scores.news2 || {};

  return (
    <motion.div className="split-screen" variants={pageVariants} initial="initial" animate="animate" exit="exit">
      {/* Top Bar */}
      <div className="split-topbar">
        <motion.button
          className="split-back-btn"
          onClick={() => navigate(`/patient/${patientId}`)}
          whileHover={{ x: -3 }}
          whileTap={{ scale: 0.96 }}
        >
          <ArrowLeft size={16} weight="bold" />
          <span>Patient Detail</span>
        </motion.button>
        <h1 className="split-title">Traditional Monitor vs Chronos Intelligence</h1>
        <span className="split-patient-id">{patientId}</span>
      </div>

      {/* Split Panels */}
      <div className="split-panels">
        {/* LEFT: Traditional Monitor */}
        <motion.div
          className="split-panel split-traditional"
          variants={detailPanelVariants}
          initial="initial"
          animate="animate"
        >
          <div className="split-panel-header split-trad-header">
            <h2>Traditional ICU Monitor</h2>
            <span className="split-panel-badge split-trad-badge">Threshold-Based</span>
          </div>

          <div className="trad-status-bar">
            {traditionalAlarmCount === 0 ? (
              <div className="trad-status trad-status-ok">
                <CheckCircle size={18} weight="fill" />
                <span>ALL PARAMETERS NORMAL</span>
              </div>
            ) : (
              <motion.div
                className="trad-status trad-status-alarm"
                animate={{ background: ['rgba(255,23,68,0.15)', 'rgba(255,23,68,0.3)', 'rgba(255,23,68,0.15)'] }}
                transition={{ duration: 1, repeat: Infinity }}
              >
                <SpeakerHigh size={18} weight="fill" />
                <span>{traditionalAlarmCount} ALARM{traditionalAlarmCount > 1 ? 'S' : ''} ACTIVE</span>
              </motion.div>
            )}
          </div>

          <div className="trad-vitals-list">
            <TraditionalVitalRow
              label="Heart Rate" icon={<Heartbeat size={16} weight="duotone" />}
              value={vitals.heart_rate?.value} unit="bpm" low={50} high={120}
            />
            <TraditionalVitalRow
              label="SpO2" icon={<Drop size={16} weight="duotone" />}
              value={vitals.spo2?.value} unit="%" low={90} high={null}
            />
            <TraditionalVitalRow
              label="Systolic BP" icon={<Drop size={16} weight="duotone" />}
              value={vitals.bp_systolic?.value} unit="mmHg" low={90} high={180}
            />
            <TraditionalVitalRow
              label="Resp Rate" icon={<Wind size={16} weight="duotone" />}
              value={vitals.resp_rate?.value} unit="/min" low={8} high={30}
            />
            <TraditionalVitalRow
              label="Temperature" icon={<Thermometer size={16} weight="duotone" />}
              value={vitals.temperature?.value} unit="C" low={35.5} high={38.5}
            />
          </div>

          <div className="trad-scores">
            <div className="trad-score-item">
              <span className="trad-score-label">NEWS2</span>
              <span className="trad-score-value">{news2.score || 0}</span>
              <span className="trad-score-risk">{news2.risk_level || 'Low'}</span>
            </div>
          </div>

          <div className="trad-assessment">
            <h3>Clinical Assessment</h3>
            <p className="trad-assessment-text">
              {traditionalAlarmCount === 0
                ? 'All vital signs within normal parameters. No intervention required. Continue routine monitoring.'
                : `${traditionalAlarmCount} parameter(s) outside normal range. Check patient.`}
            </p>
          </div>

          <div className="trad-footer">
            <span>What nurse sees today</span>
          </div>
        </motion.div>

        {/* Divider */}
        <div className="split-divider">
          <div className="split-divider-line" />
          <span className="split-divider-label">VS</span>
          <div className="split-divider-line" />
        </div>

        {/* RIGHT: Chronos Intelligence */}
        <motion.div
          className={`split-panel split-chronos split-chronos-${severity.toLowerCase()}`}
          style={{ '--split-severity-color': config.color }}
          variants={detailPanelVariants}
          initial="initial"
          animate="animate"
        >
          <div className="split-panel-header split-chronos-header">
            <h2>Chronos Entropy Intelligence</h2>
            <motion.span
              className="split-panel-badge split-chronos-badge"
              style={{ color: config.color, borderColor: config.color }}
              animate={severity === 'CRITICAL' ? { scale: [1, 1.05, 1] } : {}}
              transition={{ duration: 1, repeat: Infinity }}
            >
              {config.label}
            </motion.span>
          </div>

          <div className="chronos-ces-section">
            <CESGauge value={ces} rawValue={patient.composite_entropy_raw} severity={severity} config={config} />
          </div>

          <div className="chronos-entropy-section">
            <EntropyBars
              vitals={vitals}
              contributingVitals={alert.contributing_vitals || []}
              severityColor={config.color}
            />
          </div>

          {alert.active && (
            <motion.div
              className="chronos-alert-section"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="chronos-alert-header">
                <WarningOctagon size={16} weight="fill" style={{ color: config.color }} />
                <span style={{ color: config.color }}>{config.label} Alert</span>
              </div>
              <p className="chronos-alert-msg">{alert.message}</p>
              {alert.hours_to_predicted_event != null && (
                <div className="chronos-alert-time">
                  <Clock size={12} weight="bold" />
                  <span>Predicted event in ~{alert.hours_to_predicted_event.toFixed(1)}h</span>
                </div>
              )}
              {alert.drug_masked && (
                <div className="chronos-drug-mask-warning">
                  Drug masking detected — medications may be hiding deterioration
                </div>
              )}
            </motion.div>
          )}

          {narrative && narrative.sections?.vital_assessment && (
            <div className="chronos-narrative-section">
              <h3>AI Assessment</h3>
              <p>{narrative.sections.vital_assessment}</p>
            </div>
          )}

          {patient.interventions && patient.interventions.length > 0 && (
            <div className="chronos-interventions-section">
              <h3>
                <Lightning size={14} weight="duotone" style={{ color: 'var(--accent-teal)' }} />
                Top Recommendation
              </h3>
              <div className="chronos-top-intervention">
                <span className="chronos-intv-action">{patient.interventions[0].action}</span>
                <span className="chronos-intv-rate">
                  {(patient.interventions[0].historical_success_rate * 100).toFixed(0)}% success
                </span>
              </div>
            </div>
          )}

          <div className="chronos-footer">
            <span>What Chronos reveals</span>
          </div>
        </motion.div>
      </div>

      {/* Bottom comparison stats */}
      <motion.div className="split-comparison-bar" variants={detailPanelVariants}>
        <div className="split-comp-stat">
          <span className="split-comp-label">Traditional Alarms</span>
          <span className="split-comp-value split-comp-trad">{traditionalAlarmCount}</span>
        </div>
        <div className="split-comp-stat">
          <span className="split-comp-label">Chronos Severity</span>
          <span className="split-comp-value" style={{ color: config.color }}>{config.label}</span>
        </div>
        <div className="split-comp-stat">
          <span className="split-comp-label">Entropy Score</span>
          <AnimatedNumber value={ces} decimals={3} className="split-comp-value" style={{ color: config.color }} />
        </div>
        <div className="split-comp-stat">
          <span className="split-comp-label">NEWS2 Score</span>
          <span className="split-comp-value split-comp-trad">{news2.score || 0} ({news2.risk_level || 'Low'})</span>
        </div>
        {alert.hours_to_predicted_event != null && (
          <div className="split-comp-stat">
            <span className="split-comp-label">Predicted Event</span>
            <span className="split-comp-value" style={{ color: config.color }}>
              ~{alert.hours_to_predicted_event.toFixed(1)}h
            </span>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
}

export default SplitScreenView;
