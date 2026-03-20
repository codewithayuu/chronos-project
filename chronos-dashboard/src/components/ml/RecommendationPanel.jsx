import React from 'react';
import { Stethoscope, Flask, Users, Clock } from '@phosphor-icons/react';
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
        <div style={{ padding: '0 8px' }}>
          <h3 className="recommendation-section-title">
            <Stethoscope size={24} weight="duotone" style={{ color: 'var(--color-ml-accent)' }} /> 
            &nbsp;Interventions (from similar cases)
          </h3>
          <div className="recommendation-cards">
            {interventions.map((inv, idx) => {
              const successPct = Math.round(inv.historical_success_rate * 100);
              return (
                <div key={idx} className="intervention-card-ml">
                  <div className="intervention-rank-ml">
                    <span className="intervention-rank-number-ml">#{inv.rank}</span>
                  </div>
                  <div className="intervention-details-ml">
                    <p className="intervention-action-ml">{inv.action}</p>
                    
                    <div className="intervention-success-row-ml">
                      <div className="intervention-success-bar-track-ml">
                        <div 
                          className="intervention-success-bar-fill-ml" 
                          style={{ width: `${successPct}%`, background: successPct > 70 ? 'var(--color-none)' : 'var(--color-watch)' }} 
                        />
                      </div>
                      <span className="intervention-success-pct-ml" style={{ color: successPct > 70 ? 'var(--color-none)' : 'var(--color-watch)' }}>{successPct}%</span>
                    </div>

                    <div className="intervention-meta-ml">
                      <span className="intervention-meta-item-ml">
                        <Users size={14} weight="bold" /> n={inv.similar_cases_count}
                      </span>
                      <span className="intervention-meta-item-ml">
                        <Clock size={14} weight="bold" /> {inv.median_response_time_hours}h median
                      </span>
                    </div>
                    
                    <p className="intervention-source-ml">
                      ✨ {inv.evidence_source}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Suggested Tests based on Syndrome Pattern */}
      {suggested_tests.length > 0 && (
        <div style={{ marginTop: '24px', padding: '0 8px' }}>
          <h3 className="recommendation-section-title">
            <Flask size={24} weight="duotone" style={{ color: 'var(--color-yellow)' }} /> 
            &nbsp;Suggested Tests (pattern-based)
          </h3>
          <ul className="tests-list-ml">
            {suggested_tests.map((test, idx) => (
              <li key={idx} className="test-item-ml">
                <strong className="test-name-ml">{test.test}</strong> 
                <span className="test-reason-ml">&mdash; {test.reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="recommendation-footer-ml">
        ⚕️ Decision support only. Final decisions remain with treating physician.
      </div>
    </div>
  );
}

export default React.memo(RecommendationPanel);
