import React from 'react';
import './MLStyles.css';

const SEVERITY_COLORS = {
  'NONE': 'var(--color-green)',
  'WATCH': 'var(--color-yellow)',
  'WARNING': 'var(--color-orange)',
  'CRITICAL': 'var(--color-red)'
};

function FusionScoreGauge({ score, severity, timeEstimate }) {
  const safeScore = score || 0;
  const color = SEVERITY_COLORS[severity] || SEVERITY_COLORS['NONE'];
  const p = safeScore / 100;

  // Arc math logic
  const rad = 40;
  const circum = 2 * Math.PI * rad;
  const dashoffset = circum * (1 - (p * 0.75));

  return (
    <div className="fusion-gauge-container">
      <div style={{ position: 'relative' }}>
        <svg className="gauge-svg" viewBox="0 0 100 100">
          <circle 
            className="gauge-bg"
            cx="50" cy="50" r={rad}
            strokeDasharray={`${circum * 0.75} ${circum * 0.25}`} 
            strokeDashoffset={circum * 0.375}
            transform="rotate(135 50 50)"
          />
          <circle 
            className="gauge-fill"
            stroke={color}
            cx="50" cy="50" r={rad}
            strokeDasharray={circum}
            strokeDashoffset={dashoffset}
            transform="rotate(135 50 50)"
          />
          <text 
            x="50" y="55" 
            textAnchor="middle" 
            dominantBaseline="middle" 
            className="gauge-score"
            fill="var(--text-primary)"
          >
            {safeScore}
          </text>
        </svg>
      </div>
      <div className="gauge-severity" style={{ color }}>{severity}</div>
      <div className="gauge-time">{timeEstimate}</div>
    </div>
  );
}

export default React.memo(FusionScoreGauge);
