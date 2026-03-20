import React from 'react';
import './MLStyles.css';

const SYNDROME_MAP = {
  'Sepsis-like': { abbr: 'Sepsis', icon: '🦠', color: 'var(--color-orange)' },
  'Respiratory Failure': { abbr: 'Resp', icon: '🫁', color: 'var(--color-yellow)' },
  'Hemodynamic Instability': { abbr: 'Hemo', icon: '🫀', color: 'var(--color-red)' },
  'Cardiac Instability': { abbr: 'Cardiac', icon: '💔', color: 'var(--color-red)' },
  'Stable': null
};

function SyndromeTag({ syndromeName, confidence }) {
  if (!syndromeName || confidence < 0.40) return null;

  const mapping = SYNDROME_MAP[syndromeName];
  if (!mapping) return null;

  return (
    <span 
      className="syndrome-tag" 
      style={{ 
        background: `color-mix(in srgb, ${mapping.color} 20%, transparent)`,
        color: mapping.color,
        border: `1px solid color-mix(in srgb, ${mapping.color} 40%, transparent)`
      }}
      title={`${syndromeName} (${(confidence * 100).toFixed(0)}% confidence)`}
    >
      [{mapping.icon} {mapping.abbr}]
    </span>
  );
}

export default React.memo(SyndromeTag);
