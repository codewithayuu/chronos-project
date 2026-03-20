import React from 'react';
import './MLStyles.css';

function WarmupIndicator({ currentPoints, totalPoints }) {
  const safeCurrent = currentPoints || 0;
  const safeTotal = totalPoints || 300;
  const pct = Math.min((safeCurrent / safeTotal) * 100, 100);
  
  const remainingPoints = Math.max(safeTotal - safeCurrent, 0);
  const remainingHours = Math.floor(remainingPoints / 60);
  const remainingMins = remainingPoints % 60;

  return (
    <div className="warmup-indicator">
      <span className="warmup-text">⏳ Entropy Calibrating: {pct.toFixed(0)}%</span>
      <div className="warmup-bar-track">
        <div className="warmup-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="warmup-sub">
        Estimated: {remainingHours}h {remainingMins}m remaining
      </span>
      <span className="warmup-sub">
        Full analysis available after calibration
      </span>
    </div>
  );
}

export default React.memo(WarmupIndicator);
