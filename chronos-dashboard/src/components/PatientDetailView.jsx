import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ArrowLeft,
  Pill,
  Heartbeat,
  Drop,
  Wind,
  Thermometer,
  WarningOctagon,
  ShieldWarning,
  Stethoscope,
  Clock,
  Syringe,
} from '@phosphor-icons/react';
import VitalChart from './VitalChart';
import CESGauge from './CESGauge';
import CESHistoryChart from './CESHistoryChart';
import EntropyBars from './EntropyBars';
import InterventionCard from './InterventionCard';
import DrugBadge from './DrugBadge';
import AlertBanner from './AlertBanner';
import NarrativePanel from './NarrativePanel';
import ClinicalScorePanel from './ClinicalScorePanel';
import CorrelationPanel from './CorrelationPanel';
import DigitalTwinWrapper from './DigitalTwinWrapper';
import DrugSimulationPanel from './DrugSimulationPanel';
import { API_BASE, SEVERITY_CONFIG } from '../utils/constants';
import { formatTime } from '../utils/helpers';
import {
  pageVariants,
  detailColumnVariants,
  detailPanelVariants,
  chartStackVariants,
  chartItemVariants,
  interventionListVariants,
  interventionItemVariants,
  SPRING_SNAPPY,
} from '../utils/animations';
import './PatientDetailView.css';

const TIME_RANGES = [
  { label: '1h', hours: 1 },
  { label: '2h', hours: 2 },
  { label: '6h', hours: 6 },
  { label: '12h', hours: 12 },
];

function PatientDetailView({ patients }) {
  const { patientId } = useParams();
  const navigate = useNavigate();

  const [history, setHistory] = useState(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState(null);
  const [selectedRange, setSelectedRange] = useState(6);

  const patient = patients[patientId];

  const fetchHistory = useCallback(async (hours) => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const res = await fetch(
        `${API_BASE}/api/v1/patients/${patientId}/history?hours=${hours}` 
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setHistory(data);
    } catch (err) {
      console.error('[API] History fetch error:', err);
      setHistoryError(err.message);
    } finally {
      setHistoryLoading(false);
    }
  }, [patientId]);

  useEffect(() => {
    if (patientId) {
      fetchHistory(selectedRange);
    }
  }, [patientId, selectedRange, fetchHistory]);

  useEffect(() => {
    const interval = setInterval(() => {
      fetchHistory(selectedRange);
    }, 30000);
    return () => clearInterval(interval);
  }, [selectedRange, fetchHistory]);

  const chartData = useMemo(() => {
    if (!history || !Array.isArray(history) || history.length === 0) return null;
    return history.map((state) => ({
      time: state.timestamp,
      hr_value: state.vitals?.heart_rate?.value ?? null,
      hr_entropy: state.vitals?.heart_rate?.sampen_normalized ?? null,
      spo2_value: state.vitals?.spo2?.value ?? null,
      spo2_entropy: state.vitals?.spo2?.sampen_normalized ?? null,
      bp_sys_value: state.vitals?.bp_systolic?.value ?? null,
      bp_sys_entropy: state.vitals?.bp_systolic?.sampen_normalized ?? null,
      rr_value: state.vitals?.resp_rate?.value ?? null,
      rr_entropy: state.vitals?.resp_rate?.sampen_normalized ?? null,
      temp_value: state.vitals?.temperature?.value ?? null,
      temp_entropy: state.vitals?.temperature?.sampen_normalized ?? null,
      ces: state.composite_entropy,
      ces_raw: state.composite_entropy_raw,
    }));
  }, [history]);

  const drugEvents = useMemo(() => {
    if (!patient?.active_drugs) return [];
    return patient.active_drugs
      .filter((d) => d.start_time)
      .map((d) => ({
        time: d.start_time,
        label: `${d.drug_name} ${d.dose} ${d.unit}`,
        drugName: d.drug_name,
      }));
  }, [patient]);

  const contributingVitals = useMemo(() => {
    if (!patient?.alert?.contributing_vitals) return [];
    return patient.alert.contributing_vitals;
  }, [patient]);

  if (!patient) {
    return (
      <motion.div
        className="detail-loading-state"
        variants={pageVariants}
        initial="initial"
        animate="animate"
        exit="exit"
      >
        <div className="detail-loading-inner">
          <motion.div
            className="detail-loading-spinner"
            animate={{ rotate: 360 }}
            transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}
          />
          <h2 className="detail-loading-title">Loading patient data</h2>
          <p className="detail-loading-desc">
            Waiting for data stream for <code>{patientId}</code>
          </p>
          <motion.button
            className="detail-back-btn-loading"
            onClick={() => navigate('/')}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
          >
            <ArrowLeft size={14} weight="bold" />
            Back to Ward
          </motion.button>
        </div>
      </motion.div>
    );
  }

  const severity = patient.alert?.severity || 'NONE';
  const config = SEVERITY_CONFIG[severity] || SEVERITY_CONFIG.NONE;
  const vitals = patient.vitals || {};
  const drugs = patient.active_drugs || [];
  const interventions = patient.interventions || [];
  const alert = patient.alert || {};

  return (
    <motion.div
      className="patient-detail"
      style={{ '--detail-severity-color': config.color }}
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
    >
      {/* Top Navigation Bar */}
      <motion.div className="detail-topbar" variants={detailPanelVariants}>
        <motion.button
          className="detail-back-btn"
          onClick={() => navigate('/')}
          whileHover={{ x: -3, scale: 1.02 }}
          whileTap={{ scale: 0.96 }}
          transition={{ ...SPRING_SNAPPY }}
        >
          <ArrowLeft size={16} weight="bold" />
          <span>Ward Overview</span>
        </motion.button>

        <div className="detail-patient-info">
          <motion.span
            className="detail-severity-dot"
            style={{ background: config.color }}
            animate={{
              boxShadow: [
                `0 0 4px ${config.color}40`,
                `0 0 12px ${config.color}60`,
                `0 0 4px ${config.color}40`,
              ],
            }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          />
          <h1 className="detail-patient-id">{patientId}</h1>
          <motion.span
            className="detail-severity-badge"
            style={{
              background: `color-mix(in srgb, ${config.color} 12%, transparent)`,
              color: config.color,
              borderColor: `color-mix(in srgb, ${config.color} 25%, transparent)`,
            }}
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ ...SPRING_SNAPPY, delay: 0.15 }}
          >
            {config.label}
          </motion.span>
          {drugs.length > 0 && (
            <motion.span
              className="detail-drug-count"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ ...SPRING_SNAPPY, delay: 0.2 }}
            >
              <Pill size={12} weight="duotone" />
              {drugs.length}
            </motion.span>
          )}
        </div>

        <div className="detail-topbar-right">
          <motion.button
            className="detail-compare-btn"
            onClick={() => navigate(`/patient/${patientId}/compare`)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            transition={{ type: 'spring', stiffness: 400, damping: 30 }}
          >
            Split View
          </motion.button>
          <span className="detail-last-update">
            <Clock size={12} weight="bold" />
            {formatTime(patient.timestamp)}
          </span>
        </div>
      </motion.div>

      {/* Alert Banner */}
      {alert.active && (
        <AlertBanner alert={alert} severity={severity} config={config} />
      )}

      {/* Main Content Grid */}
      <div className="detail-content">
        {/* Left Column: Vital Charts */}
        <motion.div
          className="detail-charts-column"
          variants={detailColumnVariants}
          initial="initial"
          animate="animate"
        >
          <motion.div className="detail-section-header" variants={detailPanelVariants}>
            <div className="detail-section-title-group">
              <h2 className="detail-section-title">Vital Signs</h2>
              <span className="detail-section-subtitle">
                Value trends with entropy overlay
              </span>
            </div>

            <div className="detail-time-selector">
              {TIME_RANGES.map((range) => (
                <motion.button
                  key={range.hours}
                  className={`detail-time-btn ${selectedRange === range.hours ? 'is-active' : ''}`}
                  onClick={() => setSelectedRange(range.hours)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.92 }}
                  transition={{ ...SPRING_SNAPPY }}
                >
                  {range.label}
                </motion.button>
              ))}
            </div>
          </motion.div>

          <motion.div className="detail-chart-legend" variants={detailPanelVariants}>
            <div className="legend-item">
              <span className="legend-line legend-solid" />
              <span>Vital value</span>
            </div>
            <div className="legend-item">
              <span className="legend-line legend-dashed" />
              <span>Entropy (0-1)</span>
            </div>
            <div className="legend-item">
              <span className="legend-line legend-dotted" />
              <span>Alarm threshold</span>
            </div>
            <div className="legend-item">
              <span className="legend-line legend-drug" />
              <span>Drug event</span>
            </div>
          </motion.div>

          {historyLoading && !chartData && (
            <div className="detail-charts-loading">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="chart-skeleton">
                  <div className="chart-skeleton-header">
                    <div className="skeleton-bar skeleton-title" />
                    <div className="skeleton-bar skeleton-value" />
                  </div>
                  <div className="skeleton-chart-area" />
                </div>
              ))}
            </div>
          )}

          {historyError && (
            <div className="detail-charts-error">
              <WarningOctagon size={24} weight="duotone" color="var(--color-warning)" />
              <p>Failed to load history: {historyError}</p>
              <motion.button
                className="detail-retry-btn"
                onClick={() => fetchHistory(selectedRange)}
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.96 }}
              >
                Retry
              </motion.button>
            </div>
          )}

          {chartData && (
            <motion.div
              className="detail-charts-stack"
              variants={chartStackVariants}
              initial="initial"
              animate="animate"
            >
              <motion.div variants={chartItemVariants}>
                <VitalChart
                  data={chartData}
                  valueKey="hr_value"
                  entropyKey="hr_entropy"
                  label="Heart Rate"
                  unit="bpm"
                  icon={<Heartbeat size={16} weight="duotone" />}
                  thresholdLow={50}
                  thresholdHigh={120}
                  drugEvents={drugEvents}
                  isContributing={contributingVitals.includes('heart_rate')}
                  severityColor={config.color}
                  currentValue={vitals.heart_rate?.value}
                  currentTrend={vitals.heart_rate?.trend}
                />
              </motion.div>

              <motion.div variants={chartItemVariants}>
                <VitalChart
                  data={chartData}
                  valueKey="spo2_value"
                  entropyKey="spo2_entropy"
                  label="SpO2"
                  unit="%"
                  icon={<Drop size={16} weight="duotone" />}
                  thresholdLow={90}
                  thresholdHigh={null}
                  drugEvents={drugEvents}
                  isContributing={contributingVitals.includes('spo2')}
                  severityColor={config.color}
                  currentValue={vitals.spo2?.value}
                  currentTrend={vitals.spo2?.trend}
                  domainMin={85}
                  domainMax={100}
                />
              </motion.div>

              <motion.div variants={chartItemVariants}>
                <VitalChart
                  data={chartData}
                  valueKey="bp_sys_value"
                  entropyKey="bp_sys_entropy"
                  label="BP Systolic"
                  unit="mmHg"
                  icon={<Drop size={16} weight="duotone" />}
                  thresholdLow={90}
                  thresholdHigh={180}
                  drugEvents={drugEvents}
                  isContributing={contributingVitals.includes('bp_systolic')}
                  severityColor={config.color}
                  currentValue={vitals.bp_systolic?.value}
                  currentTrend={vitals.bp_systolic?.trend}
                />
              </motion.div>

              <motion.div variants={chartItemVariants}>
                <VitalChart
                  data={chartData}
                  valueKey="rr_value"
                  entropyKey="rr_entropy"
                  label="Respiratory Rate"
                  unit="/min"
                  icon={<Wind size={16} weight="duotone" />}
                  thresholdLow={8}
                  thresholdHigh={30}
                  drugEvents={drugEvents}
                  isContributing={contributingVitals.includes('resp_rate')}
                  severityColor={config.color}
                  currentValue={vitals.resp_rate?.value}
                  currentTrend={vitals.resp_rate?.trend}
                />
              </motion.div>

              <motion.div variants={chartItemVariants}>
                <VitalChart
                  data={chartData}
                  valueKey="temp_value"
                  entropyKey="temp_entropy"
                  label="Temperature"
                  unit="C"
                  icon={<Thermometer size={16} weight="duotone" />}
                  thresholdLow={35.5}
                  thresholdHigh={38.5}
                  drugEvents={drugEvents}
                  isContributing={contributingVitals.includes('temperature')}
                  severityColor={config.color}
                  currentValue={vitals.temperature?.value}
                  currentTrend={vitals.temperature?.trend}
                  domainMin={35}
                  domainMax={40}
                />
              </motion.div>
            </motion.div>
          )}
        </motion.div>

        {/* Right Column: Entropy + Interventions */}
        <motion.div
          className="detail-sidebar-column"
          variants={detailColumnVariants}
          initial="initial"
          animate="animate"
        >
          {/* CES Gauge */}
          <motion.div className="detail-panel detail-ces-panel" variants={detailPanelVariants}>
            <div className="detail-panel-header">
              <h3 className="detail-panel-title">Composite Entropy Score</h3>
            </div>
            <CESGauge
              value={patient.composite_entropy}
              rawValue={patient.composite_entropy_raw}
              severity={severity}
              config={config}
            />
          </motion.div>

          {/* CES History Trend */}
          {chartData && chartData.length > 5 && (
            <motion.div className="detail-panel detail-ces-history-panel" variants={detailPanelVariants}>
              <CESHistoryChart
                data={chartData}
                severity={severity}
                config={config}
              />
            </motion.div>
          )}

          {/* Digital Twin (Phase 9) */}
          <DigitalTwinWrapper patientId={patientId} />

          {/* Clinical Scores (Phase 7) */}
          <ClinicalScorePanel patient={patient} />

          {/* Organ Coupling (Phase 7) */}
          <CorrelationPanel patientId={patientId} />

          {/* Clinical Narrative (Phase 7) */}
          <NarrativePanel patientId={patientId} />

          {/* Per-Vital Entropy Bars */}
          <motion.div className="detail-panel detail-entropy-panel" variants={detailPanelVariants}>
            <div className="detail-panel-header">
              <h3 className="detail-panel-title">Per-Vital Entropy</h3>
              <span className="detail-panel-subtitle">Normalized complexity (0-1)</span>
            </div>
            <EntropyBars
              vitals={vitals}
              contributingVitals={contributingVitals}
              severityColor={config.color}
            />
          </motion.div>

          {/* Drug Simulation Lab (Phase B) */}
          <DrugSimulationPanel patientId={patientId} patient={patient} />

          {/* Active Drugs */}
          {drugs.length > 0 && (
            <motion.div className="detail-panel detail-drugs-panel" variants={detailPanelVariants}>
              <div className="detail-panel-header">
                <h3 className="detail-panel-title">
                  <Syringe size={14} weight="duotone" style={{ color: 'var(--accent-teal)' }} />
                  Active Medications
                </h3>
                <span className="detail-panel-badge">{drugs.length}</span>
              </div>
              <div className="detail-drugs-list">
                {drugs.map((drug, idx) => (
                  <DrugBadge key={`${drug.drug_name}-${idx}`} drug={drug} alert={alert} />
                ))}
              </div>
            </motion.div>
          )}

          {/* Interventions */}
          {interventions.length > 0 && (
            <motion.div className="detail-panel detail-interventions-panel" variants={detailPanelVariants}>
              <div className="detail-panel-header">
                <h3 className="detail-panel-title">
                  <Stethoscope size={14} weight="duotone" style={{ color: 'var(--accent-blue)' }} />
                  Recommended Interventions
                </h3>
              </div>
              <motion.div
                className="detail-interventions-list"
                variants={interventionListVariants}
                initial="initial"
                animate="animate"
              >
                {interventions.map((intervention, idx) => (
                  <motion.div key={`intervention-${idx}`} variants={interventionItemVariants}>
                    <InterventionCard
                      intervention={intervention}
                      index={idx}
                    />
                  </motion.div>
                ))}
              </motion.div>
              <div className="detail-disclaimer">
                <ShieldWarning size={13} weight="duotone" />
                <p>
                  These suggestions are based on historical pattern analysis and are
                  provided as decision support only. All clinical decisions remain the
                  sole responsibility of the treating physician.
                </p>
              </div>
            </motion.div>
          )}
        </motion.div>
      </div>
    </motion.div>
  );
}

export default PatientDetailView;
