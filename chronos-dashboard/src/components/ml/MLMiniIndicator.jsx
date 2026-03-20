import React from 'react';
import './MLStyles.css';

const getColor = (risk) => {
  if (risk < 0.3) return 'var(--color-green)';
  if (risk < 0.5) return 'var(--color-yellow)';
  if (risk < 0.7) return 'var(--color-orange)';
  return 'var(--color-red)';
};

function MLMiniIndicator({ risk4h, available }) {
  if (!available || risk4h === null || risk4h === undefined) {
    return (
      <div className="ml-mini-indicator ml-mini-na">
        ML N/A
      </div>
    );
  }

  const riskPct = Math.round(risk4h * 100);

  return (
    <div className="ml-mini-indicator" title={`4h Risk: ${riskPct}%`}>
      <div className="ml-mini-bar">
        <div 
          className="ml-mini-fill" 
          style={{ 
            width: `${Math.min(riskPct, 100)}%`, 
            background: getColor(risk4h) 
          }} 
        />
      </div>
      <span>{riskPct}%</span>
    </div>
  );
}

export default React.memo(MLMiniIndicator);
