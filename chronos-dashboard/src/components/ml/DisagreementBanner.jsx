import React from 'react';
import { Warning, Info } from '@phosphor-icons/react';
import './MLStyles.css';

function DisagreementBanner({ disagreement }) {
  if (!disagreement) return null;

  const { type, message } = disagreement;

  if (type === 'entropy_high_ml_low') {
    return (
      <div className="disagreement-banner entropy_high_ml_low">
        <Warning size={24} weight="duotone" />
        <div className="disagreement-content">
          <p>{message}</p>
          <span className="disagreement-res">Resolution: Conservative (entropy assessment maintained)</span>
        </div>
      </div>
    );
  }

  if (type === 'entropy_low_ml_high') {
    return (
      <div className="disagreement-banner entropy_low_ml_high">
        <Info size={24} weight="duotone" />
        <div className="disagreement-content">
          <p>{message}</p>
          <span className="disagreement-res">Resolution: Elevated monitoring recommended</span>
        </div>
      </div>
    );
  }

  return null;
}

export default React.memo(DisagreementBanner);
