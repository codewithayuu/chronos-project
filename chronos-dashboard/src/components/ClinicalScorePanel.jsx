import React from 'react';
import { motion } from 'framer-motion';
import { ChartBar, Warning } from '@phosphor-icons/react';
import AnimatedNumber from './AnimatedNumber';
import { detailPanelVariants } from '../utils/animations';
import './ClinicalScorePanel.css';

function ScoreGauge({ label, score, maxScore, risk, color }) {
  const pct = maxScore > 0 ? (score / maxScore) * 100 : 0;

  return (
    <div className="score-gauge">
      <div className="score-gauge-header">
        <span className="score-gauge-label">{label}</span>
        <span className="score-gauge-risk" style={{ color }}>
          {risk}
        </span>
      </div>
      <div className="score-gauge-bar-track">
        <motion.div
          className="score-gauge-bar-fill"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          style={{
            background: `linear-gradient(90deg, ${color}, color-mix(in srgb, ${color} 50%, transparent))`,
          }}
        />
      </div>
      <div className="score-gauge-value-row">
        <AnimatedNumber
          value={score}
          decimals={0}
          duration={600}
          className="score-gauge-value"
        />
        <span className="score-gauge-max">/{maxScore}</span>
      </div>
    </div>
  );
}

function riskColor(risk) {
  switch (risk) {
    case 'High': return 'var(--color-critical)';
    case 'Medium': return 'var(--color-warning)';
    case 'Low': return 'var(--color-none)';
    default: return 'var(--text-dim)';
  }
}

function ClinicalScorePanel({ patient }) {
  const scores = patient?.clinical_scores;
  if (!scores) return null;

  const news2 = scores.news2 || {};
  const qsofa = scores.qsofa || {};
  const ces = patient.composite_entropy;
  const severity = patient.alert?.severity || 'NONE';

  const news2Risk = news2.risk_level || 'N/A';
  const qsofaRisk = qsofa.risk_level || 'N/A';

  // The key insight: entropy detects what scores miss
  const showInsight =
    (severity === 'WARNING' || severity === 'CRITICAL') &&
    (news2Risk === 'None' || news2Risk === 'Low');

  return (
    <motion.div
      className="detail-panel clinical-score-panel"
      variants={detailPanelVariants}
    >
      <div className="detail-panel-header">
        <h3 className="detail-panel-title">
          <ChartBar size={14} weight="duotone" style={{ color: 'var(--accent-teal)' }} />
          Clinical Scores
        </h3>
        <span className="detail-panel-subtitle">Standard vs Entropy</span>
      </div>

      <div className="score-gauges-row">
        <ScoreGauge
          label="NEWS2"
          score={news2.score || 0}
          maxScore={20}
          risk={news2Risk}
          color={riskColor(news2Risk)}
        />
        <ScoreGauge
          label="qSOFA"
          score={qsofa.score || 0}
          maxScore={3}
          risk={qsofaRisk}
          color={riskColor(qsofaRisk)}
        />
      </div>

      {showInsight && (
        <motion.div
          className="score-insight"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          transition={{ delay: 0.3, duration: 0.4 }}
        >
          <Warning size={14} weight="fill" color="var(--color-warning)" />
          <p>
            <strong>Traditional scores see no problem</strong> (NEWS2={news2.score || 0}, {news2Risk}).
            But Chronos entropy is <strong style={{ color: 'var(--color-warning)' }}>{ces?.toFixed(2)}</strong> ({severity}).
            Entropy detects deterioration that standard scores miss.
          </p>
        </motion.div>
      )}

      {!showInsight && severity !== 'NONE' && (
        <div className="score-comparison-note">
          Both systems detect changes. Chronos provides earlier warning.
        </div>
      )}
    </motion.div>
  );
}

export default React.memo(ClinicalScorePanel);
