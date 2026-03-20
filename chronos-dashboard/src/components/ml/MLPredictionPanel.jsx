import React from 'react';
import { Brain } from '@phosphor-icons/react';
import RiskTimelineChart from './RiskTimelineChart';
import SyndromeDisplay from './SyndromeDisplay';
import FeatureDriversList from './FeatureDriversList';
import WarmupIndicator from './WarmupIndicator';
import './MLStyles.css';

function MLPredictionPanel({ mlPredictions, warmupMode, currentPoints }) {
  if (warmupMode) {
    return (
      <div className="ml-prediction-panel">
        <WarmupIndicator currentPoints={currentPoints} totalPoints={300} />
      </div>
    );
  }

  if (!mlPredictions || !mlPredictions.deterioration_risk) {
    return (
      <div className="ml-prediction-panel">
        <div style={{ color: 'var(--text-secondary)' }}>
          ML models unavailable
        </div>
      </div>
    );
  }

  return (
    <div className="ml-prediction-panel">
      <div className="ml-panel-header">
        <h3 className="ml-panel-title">
          <Brain size={24} weight="duotone" style={{ color: 'var(--color-ml-accent)' }} /> 
          AI Risk Analysis
        </h3>
      </div>
      
      <div className="ml-panel-content-grid">
        <div className="ml-panel-left-col">
          <RiskTimelineChart risks={mlPredictions.deterioration_risk} />
          <SyndromeDisplay syndrome={mlPredictions.syndrome} />
        </div>
        
        <div className="ml-panel-right-col">
          <FeatureDriversList drivers={mlPredictions.deterioration_risk.top_drivers} />
        </div>
      </div>
      
      <div className="ml-footer">
        <span className="info-icon">ℹ️</span>
        Trained on 100 real + 700 synthetic ICU trajectories
      </div>
    </div>
  );
}

export default React.memo(MLPredictionPanel);
