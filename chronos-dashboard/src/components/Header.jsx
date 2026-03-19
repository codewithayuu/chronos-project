import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ActivityIcon as Activity,
  Bell,
  ChartBar,
  CircleDashed,
  Heartbeat,
  SpeakerHigh,
  SpeakerSlash,
  WifiHigh,
  WifiSlash,
} from '@phosphor-icons/react';
import LiveDot from './LiveDot';
import DataFlowIndicator from './DataFlowIndicator';
import { SPRING_SNAPPY } from '../utils/animations';
import './Header.css';

function Header({
  connected,
  patientCount,
  activeAlertCount,
  systemStatus,
  onAlertClick,
  patients,
  voiceMuted,
  onVoiceToggle,
}) {
  const [showReconnecting, setShowReconnecting] = useState(false);
  const wasConnected = useRef(connected);

  useEffect(() => {
    if (wasConnected.current && !connected) {
      setShowReconnecting(true);
    } else if (connected) {
      const timeout = setTimeout(() => setShowReconnecting(false), 1500);
      return () => clearTimeout(timeout);
    }
    wasConnected.current = connected;
  }, [connected]);

  return (
    <>
      <AnimatePresence>
        {showReconnecting && !connected && (
          <motion.div
            className="reconnecting-banner"
            role="alert"
            aria-live="assertive"
            initial={{ y: -40, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -40, opacity: 0 }}
            transition={{ ...SPRING_SNAPPY }}
          >
            <WifiSlash size={14} weight="bold" style={{ marginRight: 6, verticalAlign: -2 }} />
            Connection lost. Attempting to reconnect...
          </motion.div>
        )}
      </AnimatePresence>

      <header className="header" role="banner">
        <div className="header-inner">
          <div className="header-brand">
            <motion.div
              className="header-logo-mark"
              whileHover={{ scale: 1.06 }}
              whileTap={{ scale: 0.94 }}
              transition={{ ...SPRING_SNAPPY }}
              aria-hidden="true"
            >
              <Heartbeat size={20} weight="duotone" color="var(--accent-teal)" />
            </motion.div>
            <div className="header-brand-text">
              <span className="header-title">Project Chronos</span>
              <span className="header-subtitle">Entropy-Based ICU Monitoring</span>
            </div>
          </div>

          <nav className="header-stats" aria-label="System statistics">
            <div className="header-stat">
              <Activity size={14} weight="bold" color="var(--text-secondary)" aria-hidden="true" />
              <motion.span
                className="header-stat-value"
                key={patientCount}
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ ...SPRING_SNAPPY }}
                aria-label={`${patientCount} patients monitored`}
              >
                {patientCount}
              </motion.span>
              <span className="header-stat-label" aria-hidden="true">Patients</span>
            </div>

            {systemStatus && (
              <>
                <div className="header-stat-divider" aria-hidden="true" />
                <div className="header-stat">
                  <CircleDashed size={14} weight="bold" color="var(--text-secondary)" aria-hidden="true" />
                  <span
                    className="header-stat-value"
                    aria-label={`${systemStatus.total_records_processed?.toLocaleString() || 0} records processed`}
                  >
                    {systemStatus.total_records_processed?.toLocaleString() || '--'}
                  </span>
                  <span className="header-stat-label" aria-hidden="true">Records</span>
                </div>
              </>
            )}

            <div className="header-stat-divider" aria-hidden="true" />
            <DataFlowIndicator patients={patients || {}} />
          </nav>

          <div className="header-actions">
            <motion.button
              className="header-analytics-btn"
              onClick={() => window.location.href = '/analytics'}
              whileHover={{ scale: 1.06 }}
              whileTap={{ scale: 0.92 }}
              transition={{ type: 'spring', stiffness: 400, damping: 30, mass: 0.8 }}
              aria-label="Open analytics dashboard"
            >
              <ChartBar size={16} weight="duotone" />
              <span>Analytics</span>
            </motion.button>
            <motion.button
              className={`header-voice-btn ${voiceMuted ? 'is-muted' : ''}`}
              onClick={onVoiceToggle}
              whileHover={{ scale: 1.06 }}
              whileTap={{ scale: 0.92 }}
              transition={{ type: 'spring', stiffness: 400, damping: 30, mass: 0.8 }}
              aria-label={voiceMuted ? 'Unmute voice alerts' : 'Mute voice alerts'}
            >
              {voiceMuted
                ? <SpeakerSlash size={14} weight="bold" />
                : <SpeakerHigh size={14} weight="bold" />}
            </motion.button>
            <motion.div
              className={`header-connection ${connected ? 'is-connected' : 'is-disconnected'}`}
              layout
              transition={{ ...SPRING_SNAPPY }}
              role="status"
              aria-label={connected ? 'Connected to live data stream' : 'Disconnected from data stream'}
              aria-live="polite"
            >
              {connected ? (
                <WifiHigh size={14} weight="bold" aria-hidden="true" />
              ) : (
                <WifiSlash size={14} weight="bold" aria-hidden="true" />
              )}
              <span>{connected ? 'Live' : 'Offline'}</span>
              {connected && <LiveDot isActive={true} color="var(--color-none)" size={6} />}
            </motion.div>

            <motion.button
              className="header-alert-btn"
              onClick={onAlertClick}
              aria-label={`Open alert panel. ${activeAlertCount} active alert${activeAlertCount !== 1 ? 's' : ''}`}
              aria-haspopup="dialog"
              whileHover={{ scale: 1.06 }}
              whileTap={{ scale: 0.92 }}
              transition={{ ...SPRING_SNAPPY }}
            >
              <Bell size={18} weight={activeAlertCount > 0 ? 'fill' : 'regular'} aria-hidden="true" />
              <AnimatePresence>
                {activeAlertCount > 0 && (
                  <motion.span
                    className="header-alert-badge"
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0, opacity: 0 }}
                    transition={{ ...SPRING_SNAPPY }}
                    aria-hidden="true"
                  >
                    {activeAlertCount}
                  </motion.span>
                )}
              </AnimatePresence>
            </motion.button>
          </div>
        </div>
      </header>
    </>
  );
}

export default React.memo(Header);
