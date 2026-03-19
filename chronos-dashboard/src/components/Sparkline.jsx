import React, { useMemo } from 'react';

const SPARKLINE_WIDTH = 140;
const SPARKLINE_HEIGHT = 36;
const PADDING_Y = 4;

function Sparkline({ data, severity = 'NONE', width = SPARKLINE_WIDTH, height = SPARKLINE_HEIGHT }) {
  const pathData = useMemo(() => {
    if (!data || data.length < 2) return null;

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 0.01;
    const usableHeight = height - PADDING_Y * 2;

    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * width;
      const y = PADDING_Y + usableHeight - ((value - min) / range) * usableHeight;
      return { x, y };
    });

    let d = `M ${points[0].x} ${points[0].y}`;
    for (let i = 1; i < points.length; i++) {
      const prev = points[i - 1];
      const curr = points[i];
      const cpx = (prev.x + curr.x) / 2;
      d += ` C ${cpx} ${prev.y}, ${cpx} ${curr.y}, ${curr.x} ${curr.y}`;
    }

    const lastPoint = points[points.length - 1];
    const gradientPath =
      d + ` L ${lastPoint.x} ${height} L ${points[0].x} ${height} Z`;

    return { line: d, gradient: gradientPath, lastPoint };
  }, [data, width, height]);

  if (!pathData) {
    return (
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        style={{ overflow: 'visible' }}
      >
        <line
          x1={0}
          y1={height / 2}
          x2={width}
          y2={height / 2}
          stroke="var(--text-dim)"
          strokeWidth={1}
          strokeDasharray="4 4"
          opacity={0.4}
        />
      </svg>
    );
  }

  const colorMap = {
    NONE: { stroke: '#00e676', fill: 'rgba(0, 230, 118, 0.08)' },
    WATCH: { stroke: '#ffd740', fill: 'rgba(255, 215, 64, 0.08)' },
    WARNING: { stroke: '#ff6e40', fill: 'rgba(255, 110, 64, 0.08)' },
    CRITICAL: { stroke: '#ff1744', fill: 'rgba(255, 23, 68, 0.1)' },
  };

  const colors = colorMap[severity] || colorMap.NONE;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{ overflow: 'visible' }}
    >
      <defs>
        <linearGradient id={`spark-grad-${severity}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={colors.stroke} stopOpacity={0.15} />
          <stop offset="100%" stopColor={colors.stroke} stopOpacity={0} />
        </linearGradient>
      </defs>

      {/* Gradient fill */}
      <path
        d={pathData.gradient}
        fill={`url(#spark-grad-${severity})`}
      />

      {/* Line */}
      <path
        d={pathData.line}
        fill="none"
        stroke={colors.stroke}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Current value dot */}
      <circle
        cx={pathData.lastPoint.x}
        cy={pathData.lastPoint.y}
        r={2.5}
        fill={colors.stroke}
      />
      <circle
        cx={pathData.lastPoint.x}
        cy={pathData.lastPoint.y}
        r={5}
        fill={colors.stroke}
        opacity={0.2}
      />
    </svg>
  );
}

export default React.memo(Sparkline);
