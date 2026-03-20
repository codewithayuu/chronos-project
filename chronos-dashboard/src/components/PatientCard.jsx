import React from 'react';
import { motion } from 'framer-motion';
import {
  Heartbeat,
  Drop,
  Wind,
  Pill,
  Timer,
  ArrowUp,
  ArrowDown,
  ArrowRight,
} from '@phosphor-icons/react';
import EnhancedSparkline from './EnhancedSparkline';
import AnimatedNumber from './AnimatedNumber';
import MLMiniIndicator from './ml/MLMiniIndicator';
import SyndromeTag from './ml/SyndromeTag';
import WarmupIndicator from './ml/WarmupIndicator';
import { SEVERITY_CONFIG } from '../utils/constants';
import { formatVital } from '../utils/helpers';
import { SPRING_SNAPPY, severityPulse } from '../utils/animations';
import './PatientCard.css';

const TrendArrow = React.memo(function TrendArrow({ trend }) {
  const size = 10;
  const weight = 'bold';
  switch (trend) {
    case 'rising':
      return <ArrowUp size={size} weight={weight} style={{ color: 'var(--color-none)' }} />;
    case 'falling':
      return <ArrowDown size={size} weight={weight} style={{ color: 'var(--color-warning)' }} />;
    case 'stable':
    default:
      return <ArrowRight size={size} weight={weight} style={{ color: 'var(--text-secondary)' }} />;
  }
});

function PatientCard({ patient, sparklineData }) {

  const baseSeverity = patient.alert_severity || patient.alert?.severity || 'NONE';
  const severity = patient.fusion?.final_severity || baseSeverity;
  const config = SEVERITY_CONFIG[severity] || SEVERITY_CONFIG.NONE;
  const ces = patient.composite_entropy;
  const isCalibrating = patient.calibrating;
  const windowFill = patient.window_fill || 0;

  const vitals = patient.vitals || {};
  const hr = vitals.heart_rate;
  const spo2 = vitals.spo2;
  const bpSys = vitals.bp_systolic;
  const rr = vitals.resp_rate;

  const drugs = patient.active_drugs || [];
  const hoursToEvent = patient.alert?.hours_to_predicted_event;



  const bedLabel = patient.patient_id
    ? patient.patient_id.replace('HERO-', '').replace('STABLE-', 'STB-')
    : '--';

  const pulseAnimation = severityPulse[severity] || {};

  return (
    <div
      className="patient-card-outer"
      style={{
        '--card-severity-color': config.color,
        '--card-severity-bg': config.bg,
        '--card-severity-border': config.border,
      }}
    >
      <motion.div
        className={`patient-card ${config.pulseClass}`}
        animate={pulseAnimation}
        aria-label={`Patient ${patient.patient_id}. Severity: ${config.label}`}
      >
        {/* Calibrating Overlay */}
        {isCalibrating && (
          <motion.div
            className="card-calibrating-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
          >
            <div className="calibrating-content">
              <motion.span
                className="calibrating-text"
                animate={{ y: [0, -4, 0] }}
                transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
              >
                Calibrating
              </motion.span>
              <div className="calibrating-bar-track">
                <motion.div
                  className="calibrating-bar-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${(windowFill * 100).toFixed(0)}%` }}
                  transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
                />
              </div>
              <span className="calibrating-pct">
                {(windowFill * 100).toFixed(0)}%
              </span>
            </div>
          </motion.div>
        )}

        {/* Header Row */}
        <div className="card-header">
          <div className="card-id-group">
            <motion.span
              className="card-severity-dot"
              animate={{
                boxShadow: [
                  `0 0 4px color-mix(in srgb, ${config.color} 30%, transparent)`,
                  `0 0 10px color-mix(in srgb, ${config.color} 50%, transparent)`,
                  `0 0 4px color-mix(in srgb, ${config.color} 30%, transparent)`,
                ],
              }}
              transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
            />
            <span className="card-patient-id">{bedLabel}</span>
          </div>
          <span
            className="card-severity-badge"
            style={{
              background: `color-mix(in srgb, ${config.color} 12%, transparent)`,
              color: config.color,
              borderColor: `color-mix(in srgb, ${config.color} 20%, transparent)`,
            }}
          >
            {config.label}
          </span>
        </div>

        {/* CES Score */}
        <div className="card-ces-section">
          <span className="card-ces-label">Composite Entropy</span>
          <div className="card-ces-row">
            <AnimatedNumber
              value={ces}
              decimals={2}
              duration={700}
              className="card-ces-value"
              style={{ color: config.color }}
            />
            <div className="card-ces-bar-track">
              <motion.div
                className="card-ces-bar-fill"
                animate={{
                  width: `${((ces ?? 0) * 100).toFixed(1)}%`,
                }}
                transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                style={{
                  background: `linear-gradient(90deg, ${config.color}, color-mix(in srgb, ${config.color} 60%, transparent))`,
                }}
              />
            </div>
          </div>
        </div>

        {/* Sparkline */}
        <div className="card-sparkline-container">
          <EnhancedSparkline
            data={sparklineData}
            severity={severity}
            showMinMax={true}
            showCurrentDot={true}
          />
        </div>

        {/* Vitals Grid */}
        <div className="card-vitals-grid">
          <div className="card-vital-item">
            <div className="card-vital-header">
              <Heartbeat size={12} weight="duotone" />
              <span>HR</span>
            </div>
            <div className="card-vital-value-row">
              <span className="card-vital-value">{formatVital(hr?.value)}</span>
              {hr?.trend && <TrendArrow trend={hr.trend} />}
            </div>
          </div>

          <div className="card-vital-item">
            <div className="card-vital-header">
              <Wind size={12} weight="duotone" />
              <span>SpO2</span>
            </div>
            <div className="card-vital-value-row">
              <span className="card-vital-value">{formatVital(spo2?.value)}</span>
              {spo2?.trend && <TrendArrow trend={spo2.trend} />}
            </div>
          </div>

          <div className="card-vital-item">
            <div className="card-vital-header">
              <Drop size={12} weight="duotone" />
              <span>BP</span>
            </div>
            <div className="card-vital-value-row">
              <span className="card-vital-value">{formatVital(bpSys?.value)}</span>
              {bpSys?.trend && <TrendArrow trend={bpSys.trend} />}
            </div>
          </div>

          <div className="card-vital-item">
            <div className="card-vital-header">
              <Wind size={12} weight="duotone" />
              <span>RR</span>
            </div>
            <div className="card-vital-value-row">
              <span className="card-vital-value">{formatVital(rr?.value)}</span>
              {rr?.trend && <TrendArrow trend={rr.trend} />}
            </div>
          </div>
        </div>

        {/* Footer: Drugs + Time */}
        <div className="card-footer">
          {drugs.length > 0 && (
            <motion.div
              className="card-drug-tag"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ ...SPRING_SNAPPY }}
            >
              <Pill size={12} weight="duotone" />
              <span>
                {drugs.length === 1
                  ? drugs[0].drug_name
                  : `${drugs.length} active`}
              </span>
            </motion.div>
          )}

          {hoursToEvent != null && (
            <motion.div
              className="card-time-tag"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ ...SPRING_SNAPPY, delay: 0.05 }}
            >
              <Timer size={12} weight="duotone" />
              <span>~{hoursToEvent.toFixed(1)}h</span>
            </motion.div>
          )}
        </div>

        {/* NEW: Risk Stats Row (FRS + ML) */}
        {!patient.ml_predictions?.warmup_mode && (patient.fusion || (patient.ml_predictions?.deterioration_risk)) && (
          <div className="card-risk-stats-row">
            {patient.fusion && (
              <div className="card-frs">
                <span className={`frs-score severity-${patient.fusion.final_severity.toLowerCase()}`}>
                  FRS: {patient.fusion.final_risk_score}/100
                </span>
              </div>
            )}
            {patient.ml_predictions?.deterioration_risk && (
              <MLMiniIndicator 
                risk4h={patient.ml_predictions.deterioration_risk.risk_4h}
                available={true}
              />
            )}
          </div>
        )}

        {/* NEW: Syndrome + Detector Tags */}
        <div className="card-tags" style={{ display: 'flex', gap: '4px', padding: '4px 12px', flexWrap: 'wrap' }}>
          {patient.ml_predictions?.syndrome && !patient.ml_predictions?.warmup_mode && (
            <SyndromeTag 
              syndromeName={patient.ml_predictions.syndrome.primary_syndrome}
              confidence={patient.ml_predictions.syndrome.primary_confidence}
            />
          )}
          {patient.detectors?.filter(d => d.active && d.detector_name === 'drug_masking').map(d => (
            <span key="mask" className="tag tag-drug-masked">Drug-masked</span>
          ))}
        </div>

        {/* NEW: Warmup Indicator (replace FRS section during warmup) */}
        {patient.ml_predictions?.warmup_mode && (
          <div style={{ padding: '8px 12px' }}>
            <WarmupIndicator 
              currentPoints={patient.entropy?.window_size || (windowFill * 300) || 0}
              totalPoints={300}
            />
          </div>
        )}
      </motion.div>
    </div>
  );
}

export default React.memo(PatientCard);
