import React from 'react';
import { motion } from 'framer-motion';
import { Heartbeat } from '@phosphor-icons/react';
import './InitialLoader.css';

function InitialLoader() {
  return (
    <div className="initial-loader">
      <div className="initial-loader-inner">
        {/* Logo animation */}
        <motion.div
          className="initial-loader-logo"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{
            type: 'spring',
            stiffness: 100,
            damping: 15,
            delay: 0.1,
          }}
        >
          <motion.div
            className="initial-loader-logo-ring"
            animate={{ rotate: 360 }}
            transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
          />
          <div className="initial-loader-logo-icon">
            <Heartbeat size={28} weight="duotone" color="var(--accent-teal)" />
          </div>
        </motion.div>

        {/* Title */}
        <motion.h1
          className="initial-loader-title"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          Project Chronos
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          className="initial-loader-subtitle"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.45, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          Initializing entropy monitoring system
        </motion.p>

        {/* Progress dots */}
        <motion.div
          className="initial-loader-dots"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
        >
          {[0, 1, 2].map((i) => (
            <motion.span
              key={i}
              className="initial-loader-dot"
              animate={{
                opacity: [0.3, 1, 0.3],
                scale: [0.85, 1.1, 0.85],
              }}
              transition={{
                duration: 1.2,
                repeat: Infinity,
                delay: i * 0.2,
                ease: 'easeInOut',
              }}
            />
          ))}
        </motion.div>

        {/* Status items */}
        <motion.div
          className="initial-loader-status"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8, duration: 0.4 }}
        >
          <StatusItem label="WebSocket" delay={0.9} />
          <StatusItem label="Patient data" delay={1.1} />
          <StatusItem label="Entropy engine" delay={1.3} />
        </motion.div>
      </div>
    </div>
  );
}

function StatusItem({ label, delay }) {
  return (
    <motion.div
      className="initial-loader-status-item"
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
    >
      <motion.span
        className="initial-loader-status-dot"
        animate={{
          background: [
            'var(--text-dim)',
            'var(--accent-teal)',
            'var(--text-dim)',
          ],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          delay: delay - 0.8,
        }}
      />
      <span>{label}</span>
    </motion.div>
  );
}

export default InitialLoader;
