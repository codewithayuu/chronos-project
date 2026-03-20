import React from 'react';
import { CheckCircle, WarningCircle, Warning, MagnifyingGlass } from '@phosphor-icons/react';
import './MLStyles.css';

const DETECTOR_NAMES = {
  'entropy_threshold': 'Entropy Threshold',
  'silent_decline': 'Silent Decline',
  'drug_masking': 'Drug Masking',
  'respiratory_risk': 'Respiratory Risk',
  'hemodynamic': 'Hemodynamic',
  'alarm_suppression': 'Alarm Suppression',
  'recovery': 'Recovery',
  'data_quality': 'Data Quality'
};

const SEVERITY_COLORS = {
  'NONE': 'var(--color-green)',
  'WATCH': 'var(--color-yellow)',
  'WARNING': 'var(--color-orange)',
  'CRITICAL': 'var(--color-red)'
};

function DetectorStatusList({ detectors }) {
  if (!detectors) return null;

  const activeDetectors = detectors.filter((d) => d.active);

  if (activeDetectors.length === 0) {
    return (
      <div className="detector-list">
        <div className="detector-all-normal">
          <CheckCircle size={18} weight="fill" />
          All systems normal
        </div>
      </div>
    );
  }

  return (
    <div className="detector-list">
      {activeDetectors.map((d, index) => {
        const color = SEVERITY_COLORS[d.severity] || SEVERITY_COLORS['WATCH'];
        const name = DETECTOR_NAMES[d.detector_name] || d.detector_name;

        let Icon = MagnifyingGlass;
        if (d.severity === 'CRITICAL') Icon = Warning;
        if (d.severity === 'WARNING' || d.severity === 'WATCH') Icon = WarningCircle;

        return (
          <div key={index} className="detector-item" style={{ borderColor: color }}>
            <div className="detector-header">
              <Icon size={16} color={color} weight="fill" />
              <span className="detector-name" style={{ color }}>{name}</span>
            </div>
            <p className="detector-msg">{d.message}</p>
            {d.recommended_action && (
              <p className="detector-action">Action: {d.recommended_action}</p>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default React.memo(DetectorStatusList);
