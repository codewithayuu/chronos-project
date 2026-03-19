import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight } from '@phosphor-icons/react';
import AnimatedNumber from './AnimatedNumber';
import { gaugeRevealVariants, SPRING_GENTLE } from '../utils/animations';
import './CESGauge.css';

function CESGauge({ value, rawValue, severity, config }) {
  const percentage = ((value ?? 0) * 100).toFixed(1);

  // Build the arc for the gauge
  const gaugeArc = useMemo(() => {
    const radius = 72;
    const centerX = 90;
    const centerY = 82;
    const startAngle = -210;
    const endAngle = 30;
    const totalAngle = endAngle - startAngle;
    const fillAngle = startAngle + totalAngle * Math.min(value ?? 0, 1);

    const toRad = (deg) => (deg * Math.PI) / 180;

    const bgStartX = centerX + radius * Math.cos(toRad(startAngle));
    const bgStartY = centerY + radius * Math.sin(toRad(startAngle));
    const bgEndX = centerX + radius * Math.cos(toRad(endAngle));
    const bgEndY = centerY + radius * Math.sin(toRad(endAngle));

    const fillEndX = centerX + radius * Math.cos(toRad(fillAngle));
    const fillEndY = centerY + radius * Math.sin(toRad(fillAngle));

    const largeArcBg = totalAngle > 180 ? 1 : 0;
    const fillArc = fillAngle - startAngle;
    const largeArcFill = fillArc > 180 ? 1 : 0;

    const bgPath = `M ${bgStartX} ${bgStartY} A ${radius} ${radius} 0 ${largeArcBg} 1 ${bgEndX} ${bgEndY}`;
    const fillPath = `M ${bgStartX} ${bgStartY} A ${radius} ${radius} 0 ${largeArcFill} 1 ${fillEndX} ${fillEndY}`;

    return { bgPath, fillPath, fillEndX, fillEndY };
  }, [value]);

  const showDrugAdjustment = rawValue != null && Math.abs(rawValue - value) > 0.001;

  return (
    <motion.div
      className="ces-gauge"
      variants={gaugeRevealVariants}
      initial="initial"
      animate="animate"
    >
      {/* SVG Gauge */}
      <div className="ces-gauge-svg-container">
        <svg viewBox="0 0 180 120" className="ces-gauge-svg">
          <defs>
            <linearGradient id="ces-gauge-grad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor={config.color} stopOpacity={0.2} />
              <stop offset="100%" stopColor={config.color} stopOpacity={0.8} />
            </linearGradient>
            <filter id="ces-glow">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Background arc */}
          <path
            d={gaugeArc.bgPath}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth={8}
            strokeLinecap="round"
          />

          {/* Filled arc */}
          <motion.path
            d={gaugeArc.fillPath}
            fill="none"
            stroke="url(#ces-gauge-grad)"
            strokeWidth={8}
            strokeLinecap="round"
            filter="url(#ces-glow)"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1], delay: 0.3 }}
          />

          {/* End dot */}
          <motion.circle
            cx={gaugeArc.fillEndX}
            cy={gaugeArc.fillEndY}
            r={4}
            fill={config.color}
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ ...SPRING_GENTLE, delay: 0.8 }}
            style={{
              filter: `drop-shadow(0 0 4px ${config.color})`,
            }}
          />
        </svg>

        {/* Central value */}
        <div className="ces-gauge-center">
          <AnimatedNumber
            value={value}
            decimals={2}
            duration={900}
            className="ces-gauge-value"
            style={{ color: config.color }}
          />
          <span className="ces-gauge-label">CES</span>
        </div>
      </div>

      {/* Sub-metrics */}
      <div className="ces-gauge-metrics">
        {showDrugAdjustment ? (
          <>
            <div className="ces-metric-item">
              <span className="ces-metric-label">Raw</span>
              <AnimatedNumber
                value={rawValue}
                decimals={2}
                duration={700}
                className="ces-metric-value"
              />
            </div>
            <div className="ces-metric-arrow">
              <ArrowRight size={12} weight="bold" color="var(--text-dim)" />
            </div>
            <div className="ces-metric-item">
              <span className="ces-metric-label">Adjusted</span>
              <AnimatedNumber
                value={value}
                decimals={2}
                duration={700}
                className="ces-metric-value"
                style={{ color: config.color }}
              />
            </div>
          </>
        ) : (
          <div className="ces-metric-item ces-metric-single">
            <span className="ces-metric-label">Score</span>
            <span className="ces-metric-value">{percentage}%</span>
          </div>
        )}

        <div className="ces-metric-divider" />

        <div className="ces-metric-item">
          <span className="ces-metric-label">Status</span>
          <span className="ces-metric-status" style={{ color: config.color }}>
            {config.label}
          </span>
        </div>
      </div>
    </motion.div>
  );
}

export default React.memo(CESGauge);
