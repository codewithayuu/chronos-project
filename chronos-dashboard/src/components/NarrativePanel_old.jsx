import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { BookOpen, ArrowsClockwise } from '@phosphor-icons/react';
import { API_BASE } from '../utils/constants';
import { detailPanelVariants, SPRING_SNAPPY } from '../utils/animations';
import './NarrativePanel.css';

function NarrativePanel({ patientId }) {
  const [narrative, setNarrative] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchNarrative = useCallback(async () => {
    if (!patientId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${API_BASE}/api/v1/patients/${patientId}/narrative` 
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setNarrative(data);
    } catch (err) {
      console.error('[API] Narrative fetch error:', err);
      setError(err.message);
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
      <motion.div
        className="detail-panel narrative-panel"
        variants={detailPanelVariants}
      >
        <div className="detail-panel-header">
          <h3 className="detail-panel-title">
            <BookOpen size={14} weight="duotone" style={{ color: 'var(--accent-blue)' }} />
            Clinical Narrative
          </h3>
        </div>
        <div className="narrative-loading">
          <div className="narrative-skeleton" />
          <div className="narrative-skeleton narrative-skeleton-short" />
          <div className="narrative-skeleton" />
        </div>
      </motion.div>
    );
  }

  if (error) {
    return (
      <motion.div
        className="detail-panel narrative-panel"
        variants={detailPanelVariants}
      >
        <div className="detail-panel-header">
          <h3 className="detail-panel-title">
            <BookOpen size={14} weight="duotone" style={{ color: 'var(--accent-blue)' }} />
            Clinical Narrative
          </h3>
        </div>
        <p className="narrative-error">Unable to load narrative</p>
      </motion.div>
    );
  }

  if (!narrative || !narrative.sections) return null;

  const severityClass = `narrative-severity-${(narrative.severity || 'NONE').toLowerCase()}`;

  return (
    <motion.div
      className={`detail-panel narrative-panel ${severityClass}`}
      variants={detailPanelVariants}
    >
      <div className="detail-panel-header">
        <h3 className="detail-panel-title">
          <BookOpen size={14} weight="duotone" style={{ color: 'var(--accent-blue)' }} />
          Clinical Narrative
        </h3>
        <motion.button
          className="narrative-refresh-btn"
          onClick={fetchNarrative}
          whileHover={{ scale: 1.1, rotate: 90 }}
          whileTap={{ scale: 0.9 }}
          transition={{ ...SPRING_SNAPPY }}
          aria-label="Refresh narrative"
        >
          <ArrowsClockwise size={14} weight="bold" />
        </motion.button>
      </div>

      <div className="narrative-content">
        {Object.entries(narrative.sections).map(([key, text]) => (
          <div key={key} className="narrative-section">
            <span className="narrative-section-label">
              {key.replace(/_/g, ' ')}
            </span>
            <p className="narrative-section-text">{text}</p>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

export default React.memo(NarrativePanel);
