import React, { useMemo, useId } from 'react';
import { motion } from 'framer-motion';

const DEFAULT_WIDTH = 140;
const DEFAULT_HEIGHT = 38;
const PADDING_Y = 5;
const PADDING_X = 2;

const SEVERITY_COLORS = {
  NONE: { stroke: '#00e676', glow: 'rgba(0, 230, 118, 0.3)' },
  WATCH: { stroke: '#ffd740', glow: 'rgba(255, 215, 64, 0.3)' },
  WARNING: { stroke: '#ff6e40', glow: 'rgba(255, 110, 64, 0.3)' },
  CRITICAL: { stroke: '#ff1744', glow: 'rgba(255, 23, 68, 0.35)' },
};

function buildSmoothPath(points) {
  if (points.length < 2) return '';
  let d = `M ${points[0].x} ${points[0].y}`;
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1];
    const curr = points[i];
    const cpx = (prev.x + curr.x) / 2;
    d += ` C ${cpx} ${prev.y}, ${cpx} ${curr.y}, ${curr.x} ${curr.y}`;
  }
  return d;
}

function EnhancedSparkline({
  data,
  severity = 'NONE',
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
  showMinMax = false,
  showCurrentDot = true,
}) {
  const uniqueId = useId();
  const colors = SEVERITY_COLORS[severity] || SEVERITY_COLORS.NONE;

  const { linePath, gradientPath, lastPoint, minPoint, maxPoint } = useMemo(() => {
    if (!data || data.length < 2) {
      return { linePath: null, gradientPath: null, lastPoint: null, minPoint: null, maxPoint: null };
    }

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 0.01;
    const usableWidth = width - PADDING_X * 2;
    const usableHeight = height - PADDING_Y * 2;

    const pts = data.map((value, index) => {
      const x = PADDING_X + (index / (data.length - 1)) * usableWidth;
      const y = PADDING_Y + usableHeight - ((value - min) / range) * usableHeight;
      return { x, y, value };
    });

    const line = buildSmoothPath(pts);
    const last = pts[pts.length - 1];
    const gradient = line + ` L ${last.x} ${height} L ${pts[0].x} ${height} Z`;

    let minPt = pts[0];
    let maxPt = pts[0];
    pts.forEach((p) => {
      if (p.value < minPt.value) minPt = p;
      if (p.value > maxPt.value) maxPt = p;
    });

    return { linePath: line, gradientPath: gradient, lastPoint: last, minPoint: minPt, maxPoint: maxPt, points: pts };
  }, [data, width, height]);

  if (!linePath) {
    return (
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        style={{ overflow: 'visible' }}
      >
        <line
          x1={PADDING_X}
          y1={height / 2}
          x2={width - PADDING_X}
          y2={height / 2}
          stroke="var(--text-dim)"
          strokeWidth={1}
          strokeDasharray="4 4"
          opacity={0.3}
        />
        <text
          x={width / 2}
          y={height / 2 + 12}
          textAnchor="middle"
          fontSize={8}
          fill="var(--text-dim)"
          fontFamily="var(--font-mono)"
        >
          Collecting
        </text>
      </svg>
    );
  }

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{ overflow: 'visible' }}
    >
      <defs>
        {/* Gradient fill */}
        <linearGradient id={`spark-grad-${uniqueId}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={colors.stroke} stopOpacity={0.18} />
          <stop offset="60%" stopColor={colors.stroke} stopOpacity={0.06} />
          <stop offset="100%" stopColor={colors.stroke} stopOpacity={0} />
        </linearGradient>

        {/* Glow filter */}
        <filter id={`spark-glow-${uniqueId}`}>
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>

      {/* Gradient fill area */}
      <path
        d={gradientPath}
        fill={`url(#spark-grad-${uniqueId})`}
      />

      {/* Glow line (behind) */}
      <path
        d={linePath}
        fill="none"
        stroke={colors.glow}
        strokeWidth={4}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.4}
      />

      {/* Main line */}
      <path
        d={linePath}
        fill="none"
        stroke={colors.stroke}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Min/Max markers */}
      {showMinMax && minPoint && maxPoint && data.length > 10 && (
        <>
          <circle
            cx={minPoint.x}
            cy={minPoint.y}
            r={2}
            fill="none"
            stroke={colors.stroke}
            strokeWidth={0.8}
            opacity={0.5}
          />
          <circle
            cx={maxPoint.x}
            cy={maxPoint.y}
            r={2}
            fill="none"
            stroke={colors.stroke}
            strokeWidth={0.8}
            opacity={0.5}
          />
        </>
      )}

      {/* Current value dot */}
      {showCurrentDot && lastPoint && (
        <>
          {/* Outer ping */}
          <motion.circle
            cx={lastPoint.x}
            cy={lastPoint.y}
            r={6}
            fill={colors.stroke}
            animate={{ opacity: [0.3, 0], scale: [0.8, 1.8] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeOut' }}
            style={{ transformOrigin: `${lastPoint.x}px ${lastPoint.y}px` }}
          />
          {/* Inner dot */}
          <circle
            cx={lastPoint.x}
            cy={lastPoint.y}
            r={2.5}
            fill={colors.stroke}
          />
        </>
      )}
    </svg>
  );
}

export default React.memo(EnhancedSparkline);
