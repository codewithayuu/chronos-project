import React, { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import PatientCard from './PatientCard';
import SystemStatusBar from './SystemStatusBar';
import { sortPatientsBySeverity } from '../utils/helpers';
import {
  pageVariants,
  wardGridVariants,
  wardCardVariants,
} from '../utils/animations';
import './WardView.css';

function WardView({ patients, sparklines, connected, systemStatus }) {
  const navigate = useNavigate();

  const sortedPatients = useMemo(
    () => sortPatientsBySeverity(patients),
    [patients]
  );

  if (Object.keys(patients).length === 0) {
    return (
      <motion.div
        className="ward-empty-state"
        variants={pageVariants}
        initial="initial"
        animate="animate"
        exit="exit"
      >
        <div className="ward-empty-inner">
          <motion.div
            className="ward-empty-icon"
            animate={{
              y: [0, -6, 0],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          >
            <svg
              width="48"
              height="48"
              viewBox="0 0 48 48"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <rect
                x="4"
                y="8"
                width="40"
                height="32"
                rx="4"
                stroke="var(--text-dim)"
                strokeWidth="1.5"
                strokeDasharray="4 3"
              />
              <path
                d="M24 20v8M20 24h8"
                stroke="var(--text-dim)"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </svg>
          </motion.div>
          <h2 className="ward-empty-title">Waiting for patient data</h2>
          <p className="ward-empty-desc">
            Ensure the Chronos backend is running at{' '}
            <code>localhost:8000</code> and streaming data via WebSocket.
          </p>
          <div className="ward-empty-loader">
            <span className="ward-loader-dot" style={{ animationDelay: '0ms' }} />
            <span className="ward-loader-dot" style={{ animationDelay: '200ms' }} />
            <span className="ward-loader-dot" style={{ animationDelay: '400ms' }} />
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.section
      className="ward-view"
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
    >
      <motion.div className="ward-header" variants={wardCardVariants}>
        <h1 className="ward-title">ICU Ward Overview</h1>
        <p className="ward-subtitle">
          {sortedPatients.length} patients monitored — sorted by entropy severity
        </p>
      </motion.div>

      <motion.div className="ward-grid" variants={wardGridVariants}>
        {sortedPatients.map((patient, index) => (
          <motion.div
            key={patient.patient_id}
            className="ward-grid-item"
            variants={wardCardVariants}
            layout
            layoutId={`card-shell-${patient.patient_id}`}
            onClick={() => navigate(`/patient/${patient.patient_id}`)}
            style={{ cursor: 'pointer', animationDelay: `${index * 60}ms` }}
            transition={{
              layout: {
                type: 'spring',
                stiffness: 120,
                damping: 20,
                mass: 0.9,
              },
            }}
          >
            <PatientCard
              patient={patient}
              sparklineData={sparklines[patient.patient_id]}
            />
          </motion.div>
        ))}
      </motion.div>

      <SystemStatusBar
        systemStatus={systemStatus}
        connected={connected}
        patientCount={Object.keys(patients).length}
      />
    </motion.section>
  );
}

export default React.memo(WardView);
