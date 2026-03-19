import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import { formatChartTime, CHART_COLORS } from '../utils/chartHelpers';
import { detailPanelVariants } from '../utils/animations';
import './CESHistoryChart.css';

function CESTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;

  const ces = payload.find((p) => p.dataKey === 'ces');
  const cesRaw = payload.find((p) => p.dataKey === 'ces_raw');

  return (
    <div className="ces-history-tooltip">
      <span className="ces-history-tooltip-time">{formatChartTime(label)}</span>
      {ces && (
        <div className="ces-history-tooltip-row">
          <span
            className="ces-history-tooltip-dot"
            style={{ background: 'var(--accent-teal)' }}
          />
          <span className="ces-history-tooltip-label">CES</span>
          <span className="ces-history-tooltip-value">
            {ces.value?.toFixed(3)}
          </span>
        </div>
      )}
      {cesRaw && cesRaw.value != null && (
        <div className="ces-history-tooltip-row">
          <span
            className="ces-history-tooltip-dot"
            style={{ background: 'rgba(0, 191, 165, 0.4)' }}
          />
          <span className="ces-history-tooltip-label">Raw</span>
          <span className="ces-history-tooltip-value">
            {cesRaw.value?.toFixed(3)}
          </span>
        </div>
      )}
    </div>
  );
}

function CESHistoryChart({ data, severity, config }) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];
    return data.map((d) => ({
      time: d.time,
      ces: d.ces,
      ces_raw: d.ces_raw,
    }));
  }, [data]);

  const formatXTick = (timestamp) => formatChartTime(timestamp);

  const xTickInterval = useMemo(() => {
    if (!chartData) return 0;
    if (chartData.length < 30) return 0;
    if (chartData.length < 120) return Math.floor(chartData.length / 6);
    return Math.floor(chartData.length / 8);
  }, [chartData]);

  if (!chartData || chartData.length < 2) return null;

  return (
    <motion.div
      className="ces-history-chart"
      variants={detailPanelVariants}
    >
      <div className="ces-history-header">
        <h3 className="ces-history-title">Entropy Trend</h3>
        <div className="ces-history-legend">
          <div className="ces-history-legend-item">
            <span
              className="ces-history-legend-line"
              style={{ background: 'var(--accent-teal)' }}
            />
            <span>Adjusted CES</span>
          </div>
          <div className="ces-history-legend-item">
            <span
              className="ces-history-legend-line ces-history-legend-dashed"
              style={{
                background: `repeating-linear-gradient(90deg, rgba(0,191,165,0.4) 0px, rgba(0,191,165,0.4) 4px, transparent 4px, transparent 7px)`,
              }}
            />
            <span>Raw CES</span>
          </div>
        </div>
      </div>

      <div className="ces-history-body">
        <ResponsiveContainer width="100%" height={140}>
          <AreaChart
            data={chartData}
            margin={{ top: 8, right: 8, bottom: 0, left: -14 }}
          >
            <defs>
              <linearGradient id="cesHistGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={config.color} stopOpacity={0.2} />
                <stop offset="60%" stopColor={config.color} stopOpacity={0.05} />
                <stop offset="100%" stopColor={config.color} stopOpacity={0} />
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              stroke={CHART_COLORS.grid}
              vertical={false}
            />

            <XAxis
              dataKey="time"
              tickFormatter={formatXTick}
              tick={{
                fontSize: 10,
                fill: 'var(--text-dim)',
                fontFamily: 'var(--font-mono)',
              }}
              axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
              tickLine={false}
              interval={xTickInterval}
            />

            <YAxis
              domain={[0, 1]}
              tick={{
                fontSize: 10,
                fill: 'var(--text-dim)',
                fontFamily: 'var(--font-mono)',
              }}
              axisLine={false}
              tickLine={false}
              width={35}
              tickCount={5}
            />

            <Tooltip content={<CESTooltip />} />

            {/* Severity threshold lines */}
            <ReferenceLine
              y={0.55}
              stroke="rgba(0, 230, 118, 0.2)"
              strokeDasharray="6 4"
              strokeWidth={1}
            />
            <ReferenceLine
              y={0.35}
              stroke="rgba(255, 215, 64, 0.2)"
              strokeDasharray="6 4"
              strokeWidth={1}
            />
            <ReferenceLine
              y={0.2}
              stroke="rgba(255, 23, 68, 0.2)"
              strokeDasharray="6 4"
              strokeWidth={1}
            />

            {/* Raw CES line (behind) */}
            <Area
              type="monotone"
              dataKey="ces_raw"
              stroke="rgba(0, 191, 165, 0.3)"
              strokeWidth={1}
              strokeDasharray="4 3"
              fill="none"
              dot={false}
              connectNulls
              isAnimationActive={false}
            />

            {/* Adjusted CES area + line */}
            <Area
              type="monotone"
              dataKey="ces"
              stroke={config.color}
              strokeWidth={2}
              fill="url(#cesHistGrad)"
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}

export default React.memo(CESHistoryChart);
