import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { GitBranch, ArrowsClockwise, WarningOctagon } from '@phosphor-icons/react';
import { API_BASE } from '../utils/constants';
import { detailPanelVariants, SPRING_SNAPPY } from '../utils/animations';
import './CorrelationPanel.css';

function CorrelationCell({ pair }) {
  if (!pair.data_available) {
    return (
      <div className="corr-cell corr-cell-no-data">
        <span className="corr-cell-name">{pair.pair_name}</span>
        <span className="corr-cell-status">Collecting</span>
      </div>
    );
  }

  const isDecoupled = pair.decoupled;
  const current = pair.current;
  const expected = pair.expected;

  return (
    <motion.div
      className={`corr-cell ${isDecoupled ? 'corr-cell-decoupled' : 'corr-cell-coupled'}`}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ ...SPRING_SNAPPY }}
    >
      <div className="corr-cell-header">
        <span className="corr-cell-name">{pair.pair_name}</span>
        {isDecoupled && (
          <motion.span
            className="corr-cell-alert-dot"
            animate={{
              scale: [1, 1.3, 1],
              opacity: [1, 0.6, 1],
            }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        )}
      </div>

      <div className="corr-cell-bars">
        <div className="corr-bar-row">
          <span className="corr-bar-label">Expected</span>
          <div className="corr-bar-track">
            <div
              className="corr-bar-fill corr-bar-expected"
              style={{
                width: `${Math.abs(expected) * 100}%`,
                marginLeft: expected >= 0 ? '50%' : `${50 - Math.abs(expected) * 50}%`,
              }}
            />
          </div>
          <span className="corr-bar-value">{expected.toFixed(2)}</span>
        </div>
        <div className="corr-bar-row">
          <span className="corr-bar-label">Current</span>
          <div className="corr-bar-track">
            <div
              className={`corr-bar-fill ${isDecoupled ? 'corr-bar-danger' : 'corr-bar-normal'}`}
              style={{
                width: `${Math.abs(current) * 100}%`,
                marginLeft: current >= 0 ? '50%' : `${50 - Math.abs(current) * 50}%`,
              }}
            />
          </div>
          <span className="corr-bar-value" style={{
            color: isDecoupled ? 'var(--color-critical)' : 'var(--text-secondary)',
          }}>
            {current.toFixed(2)}
          </span>
        </div>
      </div>

      {isDecoupled && (
        <p className="corr-cell-meaning">
          {pair.clinical_meaning.split('.')[0]}.
        </p>
      )}
    </motion.div>
  );
}

function CorrelationPanel({ patientId }) {
  const [correlations, setCorrelations] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchCorrelations = useCallback(async () => {
    if (!patientId) return;
    setLoading(true);
    try {
      const res = await fetch(
        `${API_BASE}/api/v1/analytics/correlations/${patientId}` 
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setCorrelations(json);
    } catch (err) {
      console.error('[API] Correlations fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [patientId]);

  useEffect(() => {
    fetchCorrelations();
    const interval = setInterval(fetchCorrelations, 15000);
    return () => clearInterval(interval);
  }, [fetchCorrelations]);

  if (!correlations) return null;

  const summary = correlations.summary || {};
  const pairs = Object.values(correlations);
  const decoupledCount = summary.decoupled_count || 0;
  const totalPairs = summary.total_pairs || 0;

  return (
    <motion.div
      className="detail-panel correlation-panel"
      variants={detailPanelVariants}
    >
      <div className="detail-panel-header">
        <h3 className="detail-panel-title">
          <GitBranch size={14} weight="duotone" style={{ color: 'var(--accent-blue)' }} />
          Organ System Coupling
        </h3>
        <div className="corr-header-right">
          <span className={`corr-count ${decoupledCount > 0 ? 'corr-count-alert' : ''}`}>
            {decoupledCount}/{totalPairs}
          </span>
          <motion.button
            className="corr-refresh-btn"
            onClick={fetchCorrelations}
            whileHover={{ scale: 1.1, rotate: 90 }}
            whileTap={{ scale: 0.9 }}
            transition={{ ...SPRING_SNAPPY }}
            aria-label="Refresh correlations"
          >
            <ArrowsClockwise size={13} weight="bold" />
          </motion.button>
        </div>
      </div>

      {summary.clinical_alert && (
        <motion.div
          className="corr-alert-banner"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
        >
          <WarningOctagon size={14} weight="fill" color="var(--color-warning)" />
          <span>{summary.clinical_alert}</span>
        </motion.div>
      )}

      <div className="corr-grid">
        {pairs.map((pair, idx) => (
          <CorrelationCell key={pair.pair_name || idx} pair={pair} />
        ))}
      </div>
    </motion.div>
  );
}

export default React.memo(CorrelationPanel);
