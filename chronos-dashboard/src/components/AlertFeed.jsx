import React, { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Warning,
  WarningOctagon,
  Eye,
  Check,
  Bell,
} from '@phosphor-icons/react';
import { SEVERITY_CONFIG } from '../utils/constants';
import { formatTime } from '../utils/helpers';
import {
  alertFeedOverlayVariants,
  alertFeedPanelVariants,
  alertItemVariants,
  SPRING_SNAPPY,
} from '../utils/animations';
import './AlertFeed.css';

function AlertFeed({ alerts, isOpen, onClose, onAcknowledge }) {
  const closeButtonRef = useRef(null);
  const panelRef = useRef(null);

  // Focus trap: focus close button on open
  useEffect(() => {
    if (isOpen && closeButtonRef.current) {
      closeButtonRef.current.focus();
    }
  }, [isOpen]);

  // Trap focus within the panel
  useEffect(() => {
    if (!isOpen) return;

    const handleTab = (e) => {
      if (e.key !== 'Tab' || !panelRef.current) return;

      const focusable = panelRef.current.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      if (focusable.length === 0) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === first) {
          e.preventDefault();
          last.focus();
        }
      } else {
        if (document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    document.addEventListener('keydown', handleTab);
    return () => document.removeEventListener('keydown', handleTab);
  }, [isOpen]);

  if (!isOpen) return null;

  const active = alerts.filter((a) => !a.acknowledged);
  const acknowledged = alerts.filter((a) => a.acknowledged);

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'CRITICAL':
        return <WarningOctagon size={16} weight="fill" />;
      case 'WARNING':
        return <Warning size={16} weight="fill" />;
      case 'WATCH':
        return <Eye size={16} weight="fill" />;
      default:
        return <Bell size={16} weight="regular" />;
    }
  };

  return (
    <>
      <motion.div
        className="alert-feed-backdrop"
        onClick={onClose}
        variants={alertFeedOverlayVariants}
        initial="initial"
        animate="animate"
        exit="exit"
        aria-hidden="true"
      />

      <motion.aside
        ref={panelRef}
        className="alert-feed-panel"
        variants={alertFeedPanelVariants}
        initial="initial"
        animate="animate"
        exit="exit"
        role="dialog"
        aria-label="Alert notifications panel"
        aria-modal="true"
      >
        <div className="alert-feed-header">
          <div className="alert-feed-title-group">
            <Bell size={18} weight="duotone" color="var(--text-primary)" aria-hidden="true" />
            <h2 className="alert-feed-title" id="alert-feed-heading">Alerts</h2>
            {active.length > 0 && (
              <motion.span
                className="alert-feed-count"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ ...SPRING_SNAPPY }}
                aria-label={`${active.length} active alerts`}
              >
                {active.length} active
              </motion.span>
            )}
          </div>
          <motion.button
            ref={closeButtonRef}
            className="alert-feed-close"
            onClick={onClose}
            aria-label="Close alert panel"
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.88 }}
            transition={{ ...SPRING_SNAPPY }}
          >
            <X size={18} weight="bold" />
          </motion.button>
        </div>

        <div className="alert-feed-body" aria-labelledby="alert-feed-heading">
          {active.length === 0 && acknowledged.length === 0 && (
            <motion.div
              className="alert-feed-empty"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              role="status"
            >
              <Check size={32} weight="duotone" color="var(--color-none)" aria-hidden="true" />
              <p>No alerts recorded yet.</p>
              <span>Alerts will appear here when entropy thresholds are breached.</span>
            </motion.div>
          )}

          {active.length > 0 && (
            <section className="alert-feed-section" aria-label="Active alerts">
              <h3 className="alert-feed-section-label">Active</h3>
              <div className="alert-feed-list" role="list">
                <AnimatePresence>
                  {active.map((alert, idx) => {
                    const sev = alert.severity || 'NONE';
                    const config = SEVERITY_CONFIG[sev] || SEVERITY_CONFIG.NONE;
                    return (
                      <motion.div
                        key={alert.alert_id || `alert-${idx}`}
                        className="alert-feed-item"
                        style={{ '--alert-color': config.color }}
                        variants={alertItemVariants}
                        initial="initial"
                        animate="animate"
                        exit="exit"
                        layout
                        role="listitem"
                        aria-label={`${config.label} alert for ${alert.patient_id}`}
                      >
                        <div className="alert-item-icon" style={{ color: config.color }} aria-hidden="true">
                          {getSeverityIcon(sev)}
                        </div>
                        <div className="alert-item-content">
                          <div className="alert-item-top">
                            <span className="alert-item-severity" style={{ color: config.color }}>
                              {config.label}
                            </span>
                            <span className="alert-item-time" aria-label={`Triggered at ${formatTime(alert.timestamp)}`}>
                              {formatTime(alert.timestamp)}
                            </span>
                          </div>
                          <span className="alert-item-patient">{alert.patient_id}</span>
                          <p className="alert-item-message">{alert.message}</p>
                          {alert.alert_id && (
                            <motion.button
                              className="alert-item-ack-btn"
                              onClick={() => onAcknowledge(alert.alert_id)}
                              whileHover={{ scale: 1.03 }}
                              whileTap={{ scale: 0.95 }}
                              transition={{ ...SPRING_SNAPPY }}
                              aria-label={`Acknowledge alert for ${alert.patient_id}`}
                            >
                              <Check size={14} weight="bold" aria-hidden="true" />
                              Acknowledge
                            </motion.button>
                          )}
                        </div>
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
              </div>
            </section>
          )}

          {acknowledged.length > 0 && (
            <section className="alert-feed-section" aria-label="Acknowledged alerts">
              <h3 className="alert-feed-section-label">Acknowledged</h3>
              <div className="alert-feed-list" role="list">
                {acknowledged.map((alert, idx) => (
                  <motion.div
                    key={alert.alert_id || `ack-${idx}`}
                    className="alert-feed-item is-acknowledged"
                    variants={alertItemVariants}
                    initial="initial"
                    animate="animate"
                    layout
                    role="listitem"
                  >
                    <div className="alert-item-icon" style={{ color: 'var(--text-dim)' }} aria-hidden="true">
                      <Check size={16} weight="duotone" />
                    </div>
                    <div className="alert-item-content">
                      <div className="alert-item-top">
                        <span className="alert-item-severity" style={{ color: 'var(--text-dim)' }}>
                          Resolved
                        </span>
                        <span className="alert-item-time">
                          {formatTime(alert.timestamp)}
                        </span>
                      </div>
                      <span className="alert-item-patient">{alert.patient_id}</span>
                      <p className="alert-item-message">{alert.message}</p>
                      {alert.acknowledged_by && (
                        <span className="alert-item-acked-by">
                          Acknowledged by {alert.acknowledged_by}
                        </span>
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>
            </section>
          )}
        </div>
      </motion.aside>
    </>
  );
}

export default React.memo(AlertFeed);
