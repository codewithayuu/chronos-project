import React from 'react';
import {
  Heartbeat,
  Drop,
  Wind,
  Thermometer,
  TrendUp,
  TrendDown,
  Minus,
} from '@phosphor-icons/react';
import './EntropyBars.css';

const VITAL_MAP = [
  { key: 'heart_rate', label: 'HR', Icon: Heartbeat },
  { key: 'spo2', label: 'SpO2', Icon: Drop },
  { key: 'bp_systolic', label: 'BP Sys', Icon: Drop },
  { key: 'bp_diastolic', label: 'BP Dia', Icon: Drop },
  { key: 'resp_rate', label: 'RR', Icon: Wind },
  { key: 'temperature', label: 'Temp', Icon: Thermometer },
];

function getBarColor(value) {
  if (value == null) return 'var(--text-dim)';
  if (value < 0.2) return 'var(--color-critical)';
  if (value < 0.35) return 'var(--color-warning)';
  if (value < 0.55) return 'var(--color-watch)';
  return 'var(--color-none)';
}

function TrendIcon({ trend }) {
  const size = 11;
  const weight = 'bold';
  switch (trend) {
    case 'rising':
      return <TrendUp size={size} weight={weight} style={{ color: 'var(--color-none)' }} />;
    case 'falling':
      return <TrendDown size={size} weight={weight} style={{ color: 'var(--color-warning)' }} />;
    default:
      return <Minus size={size} weight={weight} style={{ color: 'var(--text-dim)' }} />;
  }
}

function EntropyBars({ vitals, contributingVitals = [], severityColor }) {
  return (
    <div className="entropy-bars">
      {VITAL_MAP.map(({ key, label, Icon }) => {
        const vital = vitals[key];
        const entropy = vital?.sampen_normalized;
        const trend = vital?.trend;
        const isContributing = contributingVitals.includes(key);
        const barColor = getBarColor(entropy);

        return (
          <div
            key={key}
            className={`entropy-bar-row ${isContributing ? 'is-contributing' : ''}`}
          >
            <div className="entropy-bar-label">
              <Icon size={12} weight="duotone" />
              <span>{label}</span>
            </div>

            <div className="entropy-bar-track">
              <div
                className="entropy-bar-fill"
                style={{
                  width: `${((entropy ?? 0) * 100).toFixed(1)}%`,
                  background: `linear-gradient(90deg, ${barColor}, color-mix(in srgb, ${barColor} 60%, transparent))`,
                }}
              />
              {isContributing && (
                <div
                  className="entropy-bar-contributing-marker"
                  style={{ background: severityColor }}
                />
              )}
            </div>

            <div className="entropy-bar-value">
              <span style={{ color: barColor }}>
                {entropy != null ? entropy.toFixed(2) : '--'}
              </span>
            </div>

            <div className="entropy-bar-trend">
              <TrendIcon trend={trend} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default React.memo(EntropyBars);
