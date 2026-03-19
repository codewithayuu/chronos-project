import React from 'react';
import { motion } from 'framer-motion';
import {
  WarningOctagon,
  Warning,
  Eye,
  Timer,
  ShieldWarning,
} from '@phosphor-icons/react';
import { detailPanelVariants, SPRING_SNAPPY } from '../utils/animations';
import './AlertBanner.css';

function AlertBanner({ alert, severity, config }) {
  const getIcon = () => {
    switch (severity) {
      case 'CRITICAL':
        return <WarningOctagon size={18} weight="fill" />;
      case 'WARNING':
        return <Warning size={18} weight="fill" />;
      case 'WATCH':
        return <Eye size={18} weight="fill" />;
      default:
        return null;
    }
  };

  return (
    <motion.div
      className="alert-banner"
      style={{
        '--banner-color': config.color,
        background: `linear-gradient(135deg, color-mix(in srgb, ${config.color} 8%, var(--bg-card)), color-mix(in srgb, ${config.color} 4%, var(--bg-card)))`,
        borderColor: `color-mix(in srgb, ${config.color} 20%, transparent)`,
      }}
      variants={detailPanelVariants}
      initial="initial"
      animate="animate"
    >
      <motion.div
        className="alert-banner-icon"
        style={{ color: config.color }}
        animate={{
          scale: severity === 'CRITICAL' ? [1, 1.15, 1] : [1, 1.05, 1],
        }}
        transition={{
          duration: severity === 'CRITICAL' ? 1.2 : 2,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      >
        {getIcon()}
      </motion.div>

      <div className="alert-banner-content">
        <div className="alert-banner-top">
          <span className="alert-banner-severity" style={{ color: config.color }}>
            {config.label}
          </span>

          {alert.hours_to_predicted_event != null && (
            <motion.span
              className="alert-banner-time"
              initial={{ opacity: 0, x: 8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ ...SPRING_SNAPPY, delay: 0.2 }}
            >
              <Timer size={12} weight="bold" />
              Predicted event in ~{alert.hours_to_predicted_event.toFixed(1)}h
            </motion.span>
          )}
        </div>

        <p className="alert-banner-message">{alert.message}</p>

        {alert.drug_masked && (
          <motion.div
            className="alert-banner-drug-warning"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            transition={{ delay: 0.3, duration: 0.4 }}
          >
            <ShieldWarning size={13} weight="duotone" />
            <span>
              Drug masking detected. Underlying deterioration may be hidden by active
              medications.
            </span>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}

export default React.memo(AlertBanner);
