import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  ChartBar,
  ShieldCheck,
  SpeakerHigh,
  Lightning,
  Clock,
} from '@phosphor-icons/react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { API_BASE, SEVERITY_CONFIG } from '../utils/constants';
import {
  pageVariants,
  detailPanelVariants,
  detailColumnVariants,
  SPRING_SNAPPY,
} from '../utils/animations';
import './AnalyticsDashboard.css';

const SEVERITY_COLORS = {
  NONE: '#00e676',
  WATCH: '#ffd740',
  WARNING: '#ff6e40',
  CRITICAL: '#ff1744',
};

function StatCard({ icon, label, value, sublabel, color }) {
  return (
    <motion.div
      className="analytics-stat-card"
      variants={detailPanelVariants}
      whileHover={{ y: -2, transition: { ...SPRING_SNAPPY } }}
    >
      <div className="stat-card-icon" style={{ color: color || 'var(--accent-teal)' }}>
        {icon}
      </div>
      <div className="stat-card-content">
        <span className="stat-card-value" style={{ color: color || 'var(--text-primary)' }}>
          {value}
        </span>
        <span className="stat-card-label">{label}</span>
        {sublabel && <span className="stat-card-sublabel">{sublabel}</span>}
      </div>
    </motion.div>
  );
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="analytics-tooltip">
      <span className="analytics-tooltip-label">{label}</span>
      {payload.map((entry, idx) => (
        <div key={idx} className="analytics-tooltip-row">
          <span
            className="analytics-tooltip-dot"
            style={{ background: entry.color || entry.fill }}
          />
          <span>{entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(1) : entry.value}</span>
        </div>
      ))}
    </div>
  );
}

function AnalyticsDashboard() {
  const navigate = useNavigate();
  const [validation, setValidation] = useState(null);
  const [fatigue, setFatigue] = useState(null);
  const [dashboard, setDashboard] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    try {
      const [valRes, fatRes, dashRes] = await Promise.all([
        fetch(`${API_BASE}/api/v1/analytics/validation`),
        fetch(`${API_BASE}/api/v1/analytics/alarm-fatigue`),
        fetch(`${API_BASE}/api/v1/analytics/dashboard-summary`),
      ]);

      if (valRes.ok) setValidation(await valRes.json());
      if (fatRes.ok) setFatigue(await fatRes.json());
      if (dashRes.ok) setDashboard(await dashRes.json());
    } catch (err) {
      console.error('[Analytics] Fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 30000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  const valReport = validation?.status === 'complete' ? validation.report : null;
  const valSummary = valReport?.summary || {};
  const valCases = valReport?.cases || [];

  const fatigueComparison = fatigue?.comparison || {};
  const fatigueTrad = fatigue?.traditional_monitoring || {};
  const fatigueChronos = fatigue?.chronos_monitoring || {};

  const sevDist = dashboard?.patient_severity_distribution || [];
  const entropyRanking = dashboard?.patient_entropy_ranking || [];
  const alarmComp = dashboard?.alarm_comparison || [];

  return (
    <motion.div
      className="analytics-dashboard"
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
    >
      <motion.div className="analytics-topbar" variants={detailPanelVariants}>
        <motion.button
          className="analytics-back-btn"
          onClick={() => navigate('/')}
          whileHover={{ x: -3, scale: 1.02 }}
          whileTap={{ scale: 0.96 }}
          transition={{ ...SPRING_SNAPPY }}
        >
          <ArrowLeft size={16} weight="bold" />
          <span>Ward Overview</span>
        </motion.button>
        <h1 className="analytics-page-title">
          <ChartBar size={20} weight="duotone" color="var(--accent-teal)" />
          System Analytics
        </h1>
        <div className="analytics-topbar-right">
          {loading && <span className="analytics-loading-dot" />}
        </div>
      </motion.div>

      {/* Top Stats Row */}
      <motion.div
        className="analytics-stats-row"
        variants={detailColumnVariants}
        initial="initial"
        animate="animate"
      >
        <StatCard
          icon={<ShieldCheck size={20} weight="duotone" />}
          label="Chronos Sensitivity"
          value={valSummary.chronos_sensitivity != null
            ? `${(valSummary.chronos_sensitivity * 100).toFixed(0)}%` 
            : '--'}
          sublabel={`${valSummary.hero_cases_detected_chronos || 0}/${valSummary.total_hero_cases || 0} cases detected`}
          color="var(--color-none)"
        />
        <StatCard
          icon={<Clock size={20} weight="duotone" />}
          label="Mean Lead Time"
          value={valSummary.mean_chronos_lead_minutes != null
            ? `${valSummary.mean_chronos_lead_minutes.toFixed(0)} min` 
            : '--'}
          sublabel={`vs ${valSummary.mean_traditional_lead_minutes?.toFixed(0) || '--'} min traditional`}
          color="var(--accent-blue)"
        />
        <StatCard
          icon={<SpeakerHigh size={20} weight="duotone" />}
          label="Alarm Reduction"
          value={fatigueComparison.alarm_reduction_percent != null
            ? `${fatigueComparison.alarm_reduction_percent.toFixed(0)}%` 
            : '--'}
          sublabel={`${fatigueTrad.total_threshold_alarms || 0} traditional vs ${fatigueChronos.actionable_alerts || 0} Chronos`}
          color="var(--color-watch)"
        />
        <StatCard
          icon={<Lightning size={20} weight="duotone" />}
          label="Specificity"
          value={valSummary.chronos_specificity != null
            ? `${(valSummary.chronos_specificity * 100).toFixed(0)}%` 
            : '--'}
          sublabel={`${valSummary.stable_false_alarms || 0} false alarms on ${valSummary.stable_cases_checked || 0} stable patients`}
        />
      </motion.div>

      {/* Main Content Grid */}
      <div className="analytics-grid">
        {/* Validation Results */}
        {valCases.length > 0 && (
          <motion.div className="analytics-panel analytics-panel-wide" variants={detailPanelVariants}>
            <div className="analytics-panel-header">
              <h3>Detection Timeline — Chronos vs Traditional</h3>
            </div>
            <div className="validation-cases">
              {valCases.map((c) => (
                <div key={c.case_id} className="validation-case-row">
                  <div className="val-case-id">
                    <span className="val-case-name">{c.case_id}</span>
                    <span className="val-case-type">{c.case_type}</span>
                  </div>
                  <div className="val-case-timeline">
                    <div className="val-timeline-bar">
                      {c.chronos_warning_minute != null && (
                        <div
                          className="val-marker val-marker-chronos"
                          style={{ left: `${(c.chronos_warning_minute / c.total_records) * 100}%` }}
                          title={`Chronos WARNING at min ${c.chronos_warning_minute}`}
                        >
                          <span className="val-marker-label">Chronos</span>
                        </div>
                      )}
                      {c.traditional_alarm_minute != null && (
                        <div
                          className="val-marker val-marker-traditional"
                          style={{ left: `${(c.traditional_alarm_minute / c.total_records) * 100}%` }}
                          title={`Traditional alarm at min ${c.traditional_alarm_minute}`}
                        >
                          <span className="val-marker-label">Trad</span>
                        </div>
                      )}
                      <div
                        className="val-marker val-marker-crisis"
                        style={{ left: `${(c.crisis_minute / c.total_records) * 100}%` }}
                        title={`Crisis event at min ${c.crisis_minute}`}
                      >
                        <span className="val-marker-label">Crisis</span>
                      </div>
                    </div>
                  </div>
                  <div className="val-case-stats">
                    <span className="val-lead-chronos">
                      {c.chronos_lead_minutes != null ? `${c.chronos_lead_minutes} min` : 'N/A'}
                    </span>
                    <span className="val-lead-trad">
                      {c.traditional_lead_minutes != null ? `${c.traditional_lead_minutes} min` : 'missed'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Alarm Comparison Chart */}
        <motion.div className="analytics-panel" variants={detailPanelVariants}>
          <div className="analytics-panel-header">
            <h3>Alarm Comparison</h3>
          </div>
          <div className="analytics-chart-body">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={alarmComp} margin={{ top: 8, right: 8, bottom: 4, left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 10, fill: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {alarmComp.map((entry, idx) => (
                    <Cell key={idx} fill={entry.color || 'var(--accent-teal)'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Severity Distribution */}
        <motion.div className="analytics-panel" variants={detailPanelVariants}>
          <div className="analytics-panel-header">
            <h3>Patient Severity Distribution</h3>
          </div>
          <div className="analytics-chart-body analytics-pie-body">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={sevDist}
                  dataKey="count"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={70}
                  innerRadius={35}
                  strokeWidth={0}
                >
                  {sevDist.map((entry, idx) => (
                    <Cell
                      key={idx}
                      fill={entry.color || SEVERITY_COLORS[entry.name] || '#616161'}
                    />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend
                  iconType="circle"
                  iconSize={8}
                  wrapperStyle={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Patient Entropy Ranking */}
        {entropyRanking.length > 0 && (
          <motion.div className="analytics-panel analytics-panel-wide" variants={detailPanelVariants}>
            <div className="analytics-panel-header">
              <h3>Patient Entropy Ranking</h3>
              <span className="analytics-panel-subtitle">Sorted by risk (lowest entropy first)</span>
            </div>
            <div className="entropy-ranking-list">
              {entropyRanking.map((p, idx) => {
                const config = SEVERITY_CONFIG[p.severity] || SEVERITY_CONFIG.NONE;
                const pct = ((p.entropy || 0) * 100).toFixed(1);
                return (
                  <motion.div
                    key={p.patient_id}
                    className="entropy-ranking-row"
                    whileHover={{ background: 'rgba(255,255,255,0.02)' }}
                    onClick={() => navigate(`/patient/${p.patient_id}`)}
                    style={{ cursor: 'pointer' }}
                  >
                    <span className="ranking-position">#{idx + 1}</span>
                    <span
                      className="ranking-dot"
                      style={{ background: config.color }}
                    />
                    <span className="ranking-id">{p.patient_id}</span>
                    <div className="ranking-bar-track">
                      <motion.div
                        className="ranking-bar-fill"
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{ duration: 0.8, delay: idx * 0.05, ease: [0.16, 1, 0.3, 1] }}
                        style={{
                          background: `linear-gradient(90deg, ${config.color}, color-mix(in srgb, ${config.color} 40%, transparent))`,
                        }}
                      />
                    </div>
                    <span className="ranking-value" style={{ color: config.color }}>
                      {p.entropy?.toFixed(3) || '--'}
                    </span>
                    <span className="ranking-severity" style={{ color: config.color }}>
                      {config.label}
                    </span>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}

export default AnalyticsDashboard;
