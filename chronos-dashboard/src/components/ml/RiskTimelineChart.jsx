import React from 'react';
import { WarningCircle, Warning, CheckCircle } from '@phosphor-icons/react';
import './MLStyles.css';

const getColor = (risk) => {
  if (risk < 0.3) return 'var(--color-green)';
  if (risk < 0.5) return 'var(--color-yellow)';
  if (risk < 0.7) return 'var(--color-orange)';
  return 'var(--color-red)';
};

const getConfidenceIcon = (conf) => {
  if (conf === 'high') return <CheckCircle size={14} color="var(--color-green)" weight="fill" />;
  if (conf === 'moderate') return <WarningCircle size={14} color="var(--color-yellow)" weight="fill" />;
  if (conf === 'low') return <Warning size={14} color="var(--color-red)" weight="fill" />;
  return null;
};

function RiskTimelineChart({ risks }) {
  if (!risks) return null;

  const { risk_1h, risk_4h, risk_8h, model_confidence } = risks;

  return (
    <div className="risk-timeline">
      <div className="risk-bar-container">
        <span className="risk-label">1 hour:</span>
        <div className="risk-track">
          <div className="risk-fill" style={{ width: `${(risk_1h || 0) * 100}%`, background: getColor(risk_1h) }} />
        </div>
        <span className="risk-pct">{Math.round((risk_1h || 0) * 100)}%</span>
      </div>

      <div className="risk-bar-container risk-4h-highlight">
        <span className="risk-label">4 hours:</span>
        <div className="risk-track">
          <div className="risk-fill" style={{ width: `${(risk_4h || 0) * 100}%`, background: getColor(risk_4h) }} />
        </div>
        <span className="risk-pct">{Math.round((risk_4h || 0) * 100)}%</span>
      </div>

      <div className="risk-bar-container">
        <span className="risk-label">8 hours:</span>
        <div className="risk-track">
          <div className="risk-fill" style={{ width: `${(risk_8h || 0) * 100}%`, background: getColor(risk_8h) }} />
        </div>
        <span className="risk-pct">{Math.round((risk_8h || 0) * 100)}%</span>
      </div>

      <div className="confidence-indicator">
        {getConfidenceIcon(model_confidence)} {model_confidence ? model_confidence.toUpperCase() : 'UNKNOWN'} CONFIDENCE
      </div>
    </div>
  );
}

export default React.memo(RiskTimelineChart);
