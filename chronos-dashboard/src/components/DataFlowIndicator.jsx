import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowsClockwise } from '@phosphor-icons/react';
import { detailPanelVariants } from '../utils/animations';

function DataFlowIndicator({ patients }) {
  const [ratePerSec, setRatePerSec] = useState(0);
  const lastPatients = useRef(patients);
  const countRef = useRef(0);

  // Count updates per second
  useEffect(() => {
    const currentKeys = Object.keys(patients);

    let changes = 0;
    currentKeys.forEach((key) => {
      if (
        !lastPatients.current[key] ||
        patients[key]?.timestamp !== lastPatients.current[key]?.timestamp
      ) {
        changes++;
      }
    });

    if (changes > 0) {
      countRef.current += changes;
    }
    setRatePerSec(changes / 1); // Move this line here
    lastPatients.current = patients;
  }, [patients]);

  // Calculate rate every second
  useEffect(() => {
    const interval = setInterval(() => {
      setRatePerSec(countRef.current);
      countRef.current = 0;
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="data-flow-indicator">
      <AnimatePresence mode="wait">
        <motion.div
          key={ratePerSec}
          className="data-flow-rate"
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 4 }}
          transition={{ duration: 0.15 }}
        >
          <span className="data-flow-value">{ratePerSec}</span>
        </motion.div>
      </AnimatePresence>
      <span className="data-flow-label">msg/s</span>
      {ratePerSec > 0 && (
        <motion.span
          className="data-flow-dot"
          animate={{
            opacity: [1, 0.3, 1],
            scale: [1, 0.8, 1],
          }}
          transition={{ duration: 0.6, repeat: Infinity }}
        />
      )}
    </div>
  );
}

export default React.memo(DataFlowIndicator);
