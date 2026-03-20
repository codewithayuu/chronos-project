import React from 'react';
import './MLStyles.css';

function FeatureDriversList({ drivers }) {
  if (!drivers || drivers.length === 0) return null;

  return (
    <div className="feature-drivers">
      {drivers.map((driver, idx) => (
        <div key={idx} className="driver-item">
          <span className="driver-number">{idx + 1}</span>
          <span className="driver-desc">{driver.description}</span>
          <div className="driver-bar">
            {/* Importance is typically a decimal like 0.31 */}
            <div 
              className="driver-bar-fill" 
              style={{ width: `${Math.min(driver.importance * 100 * 2, 100)}%` }} 
            />
          </div>
          <span 
            className="driver-dir"
            style={{ color: driver.direction === 'increases_risk' ? 'var(--color-red)' : 'var(--color-green)' }}
          >
            {driver.direction === 'increases_risk' ? '↑' : '↓'}
          </span>
        </div>
      ))}
    </div>
  );
}

export default React.memo(FeatureDriversList);
