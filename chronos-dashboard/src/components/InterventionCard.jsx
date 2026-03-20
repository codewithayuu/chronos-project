import React from 'react';
import { motion } from 'framer-motion';
import { Users, Clock, Lightning } from '@phosphor-icons/react';
import { SPRING_SNAPPY } from '../utils/animations';
import './InterventionCard.css';

function getSuccessColor(rate) {
  if (rate >= 0.7) return 'var(--success-high)';
  if (rate >= 0.5) return 'var(--success-mid)';
  return 'var(--success-low)';
}

function InterventionCard({ intervention, index }) {
  const successRate = intervention.historical_success_rate ?? 0;
  const successPct = (successRate * 100).toFixed(0);
  const successColor = getSuccessColor(successRate);

  return (
    <motion.div
      className="intervention-card"
      whileHover={{
        borderColor: 'rgba(255,255,255,0.12)',
        background: 'rgba(255,255,255,0.03)',
      }}
      transition={{ ...SPRING_SNAPPY }}
    >
      {/* Rank */}
      <div className="intervention-rank">
        <span className="intervention-rank-number">#{intervention.rank}</span>
      </div>

      {/* Content */}
      <div className="intervention-content">
        <p className="intervention-action">{intervention.action}</p>

        {/* Success Bar */}
        <div className="intervention-success-row">
          <div className="intervention-success-bar-track">
            <motion.div
              className="intervention-success-bar-fill"
              initial={{ width: 0 }}
              animate={{ width: `${successPct}%` }}
              transition={{
                duration: 0.8,
                delay: index * 0.1 + 0.3,
                ease: [0.16, 1, 0.3, 1],
              }}
              style={{
                background: successColor,
              }}
            />
          </div>
          <span className="intervention-success-pct" style={{ color: successColor }}>
            {successPct}%
          </span>
        </div>

        {/* Meta */}
        <div className="intervention-meta">
          <span className="intervention-meta-item">
            <Users size={11} weight="bold" />
            n={intervention.similar_cases_count}
          </span>
          <span className="intervention-meta-item">
            <Clock size={11} weight="bold" />
            {intervention.median_response_time_hours}h median
          </span>
          {intervention.evidence_source && (
            <span className="intervention-meta-item intervention-meta-source">
              <Lightning size={11} weight="bold" />
              {intervention.evidence_source}
            </span>
          )}
        </div>
      </div>
    </motion.div>
  );
}

export default React.memo(InterventionCard);
