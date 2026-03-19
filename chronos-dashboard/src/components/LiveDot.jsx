import React from 'react';
import { motion } from 'framer-motion';

function LiveDot({ isActive = true, color = 'var(--color-none)', size = 6 }) {
  if (!isActive) return null;

  return (
    <span
      style={{
        position: 'relative',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: size,
        height: size,
      }}
    >
      {/* Ping ring */}
      <motion.span
        style={{
          position: 'absolute',
          width: size * 2.5,
          height: size * 2.5,
          borderRadius: '50%',
          background: color,
        }}
        animate={{
          opacity: [0.4, 0],
          scale: [0.6, 1.4],
        }}
        transition={{
          duration: 1.8,
          repeat: Infinity,
          ease: 'easeOut',
        }}
      />
      {/* Core dot */}
      <motion.span
        style={{
          width: size,
          height: size,
          borderRadius: '50%',
          background: color,
          position: 'relative',
          zIndex: 1,
        }}
        animate={{
          opacity: [1, 0.7, 1],
          scale: [1, 0.9, 1],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />
    </span>
  );
}

export default React.memo(LiveDot);
