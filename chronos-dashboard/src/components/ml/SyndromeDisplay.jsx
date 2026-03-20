import React from 'react';
import './MLStyles.css';

const SYNDROME_ICONS = {
  'Sepsis-like': '🦠',
  'Respiratory Failure': '🫁',
  'Hemodynamic Instability': '🫀',
  'Cardiac Instability': '💔',
  'Stable': '✅'
};

function SyndromeDisplay({ syndrome }) {
  if (!syndrome || syndrome.inconclusive) {
    return (
      <div className="syndrome-display">
        <div className="syndrome-primary" style={{ color: 'var(--text-secondary)' }}>
          Pattern inconclusive
        </div>
      </div>
    );
  }

  const { primary_syndrome, primary_confidence, secondary_syndrome, secondary_confidence, disclaimer } = syndrome;

  return (
    <div className="syndrome-display">
      <div className="syndrome-primary">
        <span>{SYNDROME_ICONS[primary_syndrome] || '🩺'}</span>
        {primary_syndrome} ({(primary_confidence * 100).toFixed(0)}%)
      </div>
      
      {secondary_syndrome && secondary_confidence > 0.20 && (
        <div className="syndrome-secondary">
          <span>{SYNDROME_ICONS[secondary_syndrome] || '🩺'}</span>
          {secondary_syndrome} ({(secondary_confidence * 100).toFixed(0)}%)
        </div>
      )}
      
      {disclaimer && <div className="syndrome-disclaimer">{disclaimer}</div>}
    </div>
  );
}

export default React.memo(SyndromeDisplay);
