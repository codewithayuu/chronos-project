import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Check,
  Database,
  Lightning,
  Timer,
  Users,
  Pulse,
} from '@phosphor-icons/react';
import { SPRING_SNAPPY } from '../utils/animations';
import './SystemStatusBar.css';

function SystemStatusBar({ systemStatus, connected, patientCount }) {
  // Only show status bar if we have system status data OR we're disconnected
  if (!systemStatus && connected) return null;

  const uptimeMin = systemStatus?.uptime_seconds
    ? Math.floor(systemStatus.uptime_seconds / 60)
    : null;
  const uptimeStr = uptimeMin != null
    ? uptimeMin >= 60
      ? `${Math.floor(uptimeMin / 60)}h ${uptimeMin % 60}m` 
      : `${uptimeMin}m` 
    : '--';

  return (
    <motion.div
      className="system-status-bar"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="status-bar-inner">
        {/* Connection */}
        <div className="status-bar-item">
          <AnimatePresence mode="wait">
            {connected ? (
              <motion.div
                key="connected"
                className="status-bar-indicator is-connected"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0 }}
                transition={{ ...SPRING_SNAPPY }}
              >
                <Check size={12} weight="fill" />
                <span>System Healthy</span>
              </motion.div>
            ) : (
              <motion.div
                key="disconnected"
                className="status-bar-indicator is-disconnected"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0 }}
                transition={{ ...SPRING_SNAPPY }}
              >
                <Pulse size={12} weight="bold" />
                <span>Reconnecting</span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="status-bar-divider" />

        {/* Stats */}
        {systemStatus && (
          <>
            <div className="status-bar-item">
              <Users size={11} weight="bold" />
              <span>{systemStatus.active_patients ?? patientCount} patients</span>
            </div>

            <div className="status-bar-divider" />

            <div className="status-bar-item">
              <Database size={11} weight="bold" />
              <span>
                {systemStatus.total_records_processed?.toLocaleString() ?? '--'} records
              </span>
            </div>

            <div className="status-bar-divider" />

            <div className="status-bar-item">
              <Lightning size={11} weight="bold" />
              <span>
                {systemStatus.active_alerts ?? 0} active alerts
              </span>
            </div>

            <div className="status-bar-divider" />

            <div className="status-bar-item">
              <Timer size={11} weight="bold" />
              <span>Uptime {uptimeStr}</span>
            </div>

            {systemStatus.evidence_engine_ready && (
              <>
                <div className="status-bar-divider" />
                <div className="status-bar-item status-bar-evidence">
                  <Check size={11} weight="fill" />
                  <span>{systemStatus.evidence_cases ?? 0} evidence cases</span>
                </div>
              </>
            )}
          </>
        )}
      </div>
    </motion.div>
  );
}

export default React.memo(SystemStatusBar);
