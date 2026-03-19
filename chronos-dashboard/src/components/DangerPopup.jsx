import React, { useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
  WarningOctagon,
  Heartbeat,
  ArrowRight,
  X,
  SpeakerHigh,
  Clock,
  ShieldWarning,
} from '@phosphor-icons/react';
import AnimatedNumber from './AnimatedNumber';
import { SEVERITY_CONFIG } from '../utils/constants';
import { SPRING_SNAPPY } from '../utils/animations';
import './DangerPopup.css';

function DangerPopup({ alert, patient, onDismiss }) {
  const navigate = useNavigate();
  const audioRef = useRef(null);
  const hasPlayedSound = useRef(false);

  // Play alert sound
  useEffect(() => {
    if (!alert || hasPlayedSound.current) return;
    hasPlayedSound.current = true;

    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const severity = alert.severity;

      if (severity === 'CRITICAL') {
        // Triple beep for critical
        [0, 300, 600].forEach((delay) => {
          setTimeout(() => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.frequency.value = 880;
            osc.type = 'sine';
            gain.gain.setValueAtTime(0.3, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.2);
            osc.start(ctx.currentTime);
            osc.stop(ctx.currentTime + 0.2);
          }, delay);
        });
      } else {
        // Single beep for warning
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = 660;
        osc.type = 'sine';
        gain.gain.setValueAtTime(0.2, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.3);
      }
    } catch (e) {
      // Audio not available
    }
  }, [alert]);

  const handleGoToPatient = useCallback(() => {
    if (alert?.patient_id) {
      onDismiss();
      navigate(`/patient/${alert.patient_id}`);
    }
  }, [alert, navigate, onDismiss]);

  // Auto-dismiss after 30 seconds
  useEffect(() => {
    if (!alert) return;
    const timeout = setTimeout(onDismiss, 30000);
    return () => clearTimeout(timeout);
  }, [alert, onDismiss]);

  // Dismiss on Escape
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'Escape') onDismiss();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onDismiss]);

  if (!alert) return null;

  const severity = alert.severity || 'WARNING';
  const config = SEVERITY_CONFIG[severity] || SEVERITY_CONFIG.WARNING;
  const isCritical = severity === 'CRITICAL';
  const patientId = alert.patient_id || 'Unknown';
  const message = alert.message || 'Entropy deterioration detected';
  const drugMasked = alert.drug_masked || false;

  // Get patient vitals if available
  const vitals = patient?.vitals || {};
  const ces = patient?.composite_entropy;
  const hoursToEvent = alert.hours_to_predicted_event || patient?.alert?.hours_to_predicted_event;

  const bedLabel = patientId.replace('HERO-', '').replace('STABLE-', 'STB-');

  return (
    <motion.div
      className={`danger-popup-overlay ${isCritical ? 'danger-critical' : 'danger-warning'}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onDismiss();
      }}
    >
      <motion.div
        className="danger-popup-card"
        initial={{ scale: 0.8, y: 40, opacity: 0 }}
        animate={{ scale: 1, y: 0, opacity: 1 }}
        exit={{ scale: 0.9, y: 20, opacity: 0 }}
        transition={{ type: 'spring', stiffness: 300, damping: 25 }}
      >
        {/* Top severity bar */}
        <motion.div
          className="danger-severity-bar"
          style={{ background: config.color }}
          animate={isCritical ? {
            opacity: [1, 0.6, 1],
          } : {}}
          transition={{ duration: 0.8, repeat: Infinity }}
        />

        {/* Close button */}
        <motion.button
          className="danger-close-btn"
          onClick={onDismiss}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          <X size={18} weight="bold" />
        </motion.button>

        {/* Icon + Severity */}
        <div className="danger-header">
          <motion.div
            className="danger-icon"
            style={{ color: config.color }}
            animate={{
              scale: isCritical ? [1, 1.2, 1] : [1, 1.1, 1],
              rotate: isCritical ? [0, -5, 5, 0] : [0],
            }}
            transition={{
              duration: isCritical ? 0.8 : 2,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          >
            <WarningOctagon size={48} weight="fill" />
          </motion.div>

          <motion.h2
            className="danger-severity-text"
            style={{ color: config.color }}
            animate={isCritical ? { scale: [1, 1.02, 1] } : {}}
            transition={{ duration: 1, repeat: Infinity }}
          >
            {isCritical ? 'CRITICAL ALERT' : 'WARNING ALERT'}
          </motion.h2>
        </div>

        {/* Patient ID */}
        <div className="danger-patient-section">
          <div className="danger-bed-label">
            <Heartbeat size={16} weight="duotone" style={{ color: config.color }} />
            <span className="danger-bed-number">{bedLabel}</span>
          </div>
          <span className="danger-patient-id">{patientId}</span>
        </div>

        {/* Alert Message */}
        <p className="danger-message">{message}</p>

        {/* Key Metrics */}
        <div className="danger-metrics">
          {ces != null && (
            <div className="danger-metric">
              <span className="danger-metric-label">Entropy Score</span>
              <AnimatedNumber
                value={ces}
                decimals={3}
                className="danger-metric-value"
                style={{ color: config.color }}
              />
            </div>
          )}

          {hoursToEvent != null && (
            <div className="danger-metric">
              <span className="danger-metric-label">
                <Clock size={12} weight="bold" />
                Predicted Event
              </span>
              <span className="danger-metric-value" style={{ color: config.color }}>
                ~{hoursToEvent.toFixed(1)}h
              </span>
            </div>
          )}

          {vitals.heart_rate?.value != null && (
            <div className="danger-metric">
              <span className="danger-metric-label">Heart Rate</span>
              <span className="danger-metric-value">
                {vitals.heart_rate.value.toFixed(0)} bpm
              </span>
            </div>
          )}

          {vitals.spo2?.value != null && (
            <div className="danger-metric">
              <span className="danger-metric-label">SpO2</span>
              <span className="danger-metric-value">
                {vitals.spo2.value.toFixed(0)}%
              </span>
            </div>
          )}
        </div>

        {/* Drug Masking Warning */}
        {drugMasked && (
          <motion.div
            className="danger-drug-warning"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
          >
            <ShieldWarning size={16} weight="duotone" />
            <span>Drug masking detected — medications may be hiding true severity</span>
          </motion.div>
        )}

        {/* Action Buttons */}
        <div className="danger-actions">
          <motion.button
            className="danger-go-btn"
            onClick={handleGoToPatient}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            style={{
              background: `linear-gradient(135deg, ${config.color}, color-mix(in srgb, ${config.color} 70%, #000))`,
            }}
          >
            <span>Go to Patient</span>
            <ArrowRight size={16} weight="bold" />
          </motion.button>

          <motion.button
            className="danger-dismiss-btn"
            onClick={onDismiss}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
          >
            Dismiss
          </motion.button>
        </div>

        {/* Auto-dismiss timer */}
        <motion.div
          className="danger-timer-bar"
          initial={{ width: '100%' }}
          animate={{ width: '0%' }}
          transition={{ duration: 30, ease: 'linear' }}
          style={{ background: config.color }}
        />
      </motion.div>
    </motion.div>
  );
}

export default React.memo(DangerPopup);
