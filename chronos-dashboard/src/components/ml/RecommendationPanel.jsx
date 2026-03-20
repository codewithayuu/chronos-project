import React from 'react';
import { Stethoscope, Flask } from '@phosphor-icons/react';
import './MLStyles.css';

function RecommendationPanel({ recommendations }) {
  if (!recommendations) return null;

  const { interventions = [], suggested_tests = [] } = recommendations;

  if (interventions.length === 0 && suggested_tests.length === 0) {
    return null;
  }

  return (
    <div className="recommendation-panel">
      
      {/* Interventions from similar cases */}
      {interventions.length > 0 && (
        <div>
          <h3 className="recommendation-section-title">
            <Stethoscope size={18} weight="duotone" style={{ color: 'var(--color-ml-accent)' }} /> 
            &nbsp;Interventions (from similar cases)
          </h3>
          <div className="recommendation-cards">
            {interventions.map((inv, idx) => (
              <div key={idx} className="intervention-card-ml">
                <div className="intervention-rank">#{inv.rank}</div>
                <div className="intervention-details">
                  <p className="intervention-action">{inv.action}</p>
                  <p className="intervention-stats">
                    Success rate: {Math.round(inv.historical_success_rate * 100)}% (n={inv.similar_cases_count}) 
                    <br/>
                    Typical response: {inv.median_response_time_hours} hours
                  </p>
                  <p style={{ fontSize: '0.7em', color: 'var(--text-secondary)', margin: '4px 0 0 0', fontStyle: 'italic' }}>
                    {inv.evidence_source}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Suggested Tests based on Syndrome Pattern */}
      {suggested_tests.length > 0 && (
        <div style={{ marginTop: '12px' }}>
          <h3 className="recommendation-section-title">
            <Flask size={18} weight="duotone" style={{ color: 'var(--color-yellow)' }} /> 
            &nbsp;Suggested Tests (pattern-based)
          </h3>
          <ul className="tests-list">
            {suggested_tests.map((test, idx) => (
              <li key={idx}><strong>{test.test}</strong> &mdash; {test.reason}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="recommendation-footer">
        ⚕️ Decision support only. Final decisions remain with treating physician.
      </div>
    </div>
  );
}

export default React.memo(RecommendationPanel);
