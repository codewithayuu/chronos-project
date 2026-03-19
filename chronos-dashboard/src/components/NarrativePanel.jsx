import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BookOpen, ArrowsClockwise, CaretDown, CaretUp } from '@phosphor-icons/react';
import { API_BASE } from '../utils/constants';
import { detailPanelVariants, SPRING_SNAPPY } from '../utils/animations';
import './NarrativePanel.css';

function NarrativePanel({ patientId }) {
  const [narrative, setNarrative] = useState(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const fetchNarrative = useCallback(async () => {
    if (!patientId) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/patients/${patientId}/narrative`);
      if (res.ok) setNarrative(await res.json());
    } catch (err) {
      console.error('[API] Narrative error:', err);
    } finally {
      setLoading(false);
    }
  }, [patientId]);

  useEffect(() => {
    fetchNarrative();
    const interval = setInterval(fetchNarrative, 30000);
    return () => clearInterval(interval);
  }, [fetchNarrative]);

  if (loading && !narrative) {
    return (
      <motion.div className="detail-panel narrative-panel" variants={detailPanelVariants}>
        <div className="detail-panel-header">
          <h3 className="detail-panel-title">
            <BookOpen size={14} weight="duotone" style={{ color: 'var(--accent-blue)' }} />
            AI Clinical Summary
          </h3>
        </div>
        <div className="narrative-loading">
          <div className="narrative-skeleton" />
          <div className="narrative-skeleton narrative-skeleton-short" />
        </div>
      </motion.div>
    );
  }

  if (!narrative || !narrative.sections) return null;

  const overview = narrative.sections.overview || '';
  const riskAssessment = narrative.sections.risk_assessment || '';
  const otherSections = Object.entries(narrative.sections).filter(
    ([key]) => key !== 'overview' && key !== 'risk_assessment'
  );

  const severityClass = `narrative-severity-${(narrative.severity || 'NONE').toLowerCase()}`;

  return (
    <motion.div className={`detail-panel narrative-panel ${severityClass}`} variants={detailPanelVariants}>
      <div className="detail-panel-header">
        <h3 className="detail-panel-title">
          <BookOpen size={14} weight="duotone" style={{ color: 'var(--accent-blue)' }} />
          AI Clinical Summary
        </h3>
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
          <motion.button
            className="narrative-refresh-btn"
            onClick={fetchNarrative}
            whileHover={{ scale: 1.1, rotate: 90 }}
            whileTap={{ scale: 0.9 }}
            transition={{ ...SPRING_SNAPPY }}
          >
            <ArrowsClockwise size={14} weight="bold" />
          </motion.button>
        </div>
      </div>

      {/* Summary (always visible) */}
      <div className="narrative-summary">
        {overview && <p className="narrative-overview-text">{overview}</p>}
        {riskAssessment && <p className="narrative-risk-text">{riskAssessment}</p>}
      </div>

      {/* Expand toggle */}
      {otherSections.length > 0 && (
        <>
          <motion.button
            className="narrative-expand-btn"
            onClick={() => setExpanded(!expanded)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <span>{expanded ? 'Show less' : `Show detailed analysis (${otherSections.length} sections)`}</span>
            {expanded ? <CaretUp size={12} weight="bold" /> : <CaretDown size={12} weight="bold" />}
          </motion.button>

          <AnimatePresence>
            {expanded && (
              <motion.div
                className="narrative-details"
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                {otherSections.map(([key, text]) => (
                  <div key={key} className="narrative-section">
                    <span className="narrative-section-label">{key.replace(/_/g, ' ')}</span>
                    <p className="narrative-section-text">{text}</p>
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}
    </motion.div>
  );
}

export default React.memo(NarrativePanel);
