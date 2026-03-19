import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  Tooltip,
} from 'recharts';
import { TrendUp, TrendDown, Minus } from '@phosphor-icons/react';
import { formatVital, formatTime } from '../utils/helpers';
import './VitalChart.css';

function TrendIcon({ trend }) {
  const size = 13;
  const weight = 'bold';
  switch (trend) {
    case 'rising':
      return <TrendUp size={size} weight={weight} style={{ color: 'var(--color-none)' }} />;
    case 'falling':
      return <TrendDown size={size} weight={weight} style={{ color: 'var(--color-warning)' }} />;
    default:
      return <Minus size={size} weight={weight} style={{ color: 'var(--text-dim)' }} />;
  }
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;

  const time = formatTime(label);
  const vitalPayload = payload.find((p) => p.dataKey && !p.dataKey.includes('entropy'));
  const entropyPayload = payload.find((p) => p.dataKey && p.dataKey.includes('entropy'));

  return (
    <div className="vital-tooltip">
      <span className="vital-tooltip-time">{time}</span>
      {vitalPayload && (
        <div className="vital-tooltip-row">
          <span className="vital-tooltip-dot" style={{ background: '#4fc3f7' }} />
          <span className="vital-tooltip-value">
            {formatVital(vitalPayload.value)}
          </span>
        </div>
      )}
      {entropyPayload && entropyPayload.value != null && (
        <div className="vital-tooltip-row">
          <span className="vital-tooltip-dot" style={{ background: '#ba68c8' }} />
          <span className="vital-tooltip-value">
            {entropyPayload.value.toFixed(3)}
          </span>
          <span className="vital-tooltip-label">entropy</span>
        </div>
      )}
    </div>
  );
}

function VitalChart({
  data,
  valueKey,
  entropyKey,
  label,
  unit,
  icon,
  thresholdLow,
  thresholdHigh,
  drugEvents = [],
  isContributing = false,
  severityColor,
  currentValue,
  currentTrend,
  domainMin,
  domainMax,
}) {
  // Compute Y domain for vital values
  const valueDomain = useMemo(() => {
    if (domainMin != null && domainMax != null) return [domainMin, domainMax];
    if (!data || data.length === 0) return ['auto', 'auto'];

    const values = data.map((d) => d[valueKey]).filter((v) => v != null);
    if (values.length === 0) return ['auto', 'auto'];

    let min = Math.min(...values);
    let max = Math.max(...values);

    // Include thresholds in domain
    if (thresholdLow != null) min = Math.min(min, thresholdLow);
    if (thresholdHigh != null) max = Math.max(max, thresholdHigh);

    const padding = (max - min) * 0.1 || 5;
    return [
      domainMin != null ? domainMin : Math.floor(min - padding),
      domainMax != null ? domainMax : Math.ceil(max + padding),
    ];
  }, [data, valueKey, thresholdLow, thresholdHigh, domainMin, domainMax]);

  // Format X-axis time labels
  const formatXTick = (timestamp) => {
    if (!timestamp) return '';
    const d = new Date(timestamp);
    return d.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    });
  };

  // Reduce tick count for readability
  const xTickInterval = useMemo(() => {
    if (!data) return 0;
    if (data.length < 30) return 0;
    if (data.length < 120) return Math.floor(data.length / 6);
    return Math.floor(data.length / 8);
  }, [data]);

  return (
    <div
      className={`vital-chart-container ${isContributing ? 'is-contributing' : ''}`}
      style={{
        '--contributing-color': severityColor,
      }}
    >
      {/* Header */}
      <div className="vital-chart-header">
        <div className="vital-chart-label-group">
          {isContributing && (
            <span className="vital-chart-contributing-indicator" />
          )}
          <span className="vital-chart-icon">{icon}</span>
          <span className="vital-chart-label">{label}</span>
          {isContributing && (
            <span className="vital-chart-contributing-tag">Contributing</span>
          )}
        </div>
        <div className="vital-chart-current">
          <span className="vital-chart-current-value">
            {formatVital(currentValue)}
          </span>
          <span className="vital-chart-current-unit">{unit}</span>
          {currentTrend && <TrendIcon trend={currentTrend} />}
        </div>
      </div>

      {/* Chart */}
      <div className="vital-chart-body">
        <ResponsiveContainer width="100%" height={110}>
          <ComposedChart
            data={data}
            margin={{ top: 4, right: 8, bottom: 0, left: -14 }}
          >
            <defs>
              <linearGradient id={`grad-${valueKey}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#4fc3f7" stopOpacity={0.12} />
                <stop offset="100%" stopColor="#4fc3f7" stopOpacity={0} />
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.04)"
              vertical={false}
            />

            <XAxis
              dataKey="time"
              tickFormatter={formatXTick}
              tick={{ fontSize: 10, fill: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
              tickLine={false}
              interval={xTickInterval}
            />

            {/* Left Y-Axis: Vital value */}
            <YAxis
              yAxisId="value"
              domain={valueDomain}
              tick={{ fontSize: 10, fill: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}
              axisLine={false}
              tickLine={false}
              width={40}
            />

            {/* Right Y-Axis: Entropy (0-1) */}
            <YAxis
              yAxisId="entropy"
              orientation="right"
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: 'rgba(186, 104, 200, 0.5)', fontFamily: 'var(--font-mono)' }}
              axisLine={false}
              tickLine={false}
              width={30}
              tickCount={3}
            />

            <Tooltip content={<CustomTooltip />} />

            {/* Threshold lines */}
            {thresholdLow != null && (
              <ReferenceLine
                yAxisId="value"
                y={thresholdLow}
                stroke="var(--chart-threshold)"
                strokeDasharray="3 3"
                strokeWidth={1}
              />
            )}
            {thresholdHigh != null && (
              <ReferenceLine
                yAxisId="value"
                y={thresholdHigh}
                stroke="var(--chart-threshold)"
                strokeDasharray="3 3"
                strokeWidth={1}
              />
            )}

            {/* Drug event markers */}
            {drugEvents.map((de, i) => (
              <ReferenceLine
                key={`drug-${i}`}
                yAxisId="value"
                x={de.time}
                stroke="var(--chart-drug-marker)"
                strokeDasharray="5 3"
                strokeWidth={1}
                label={{
                  value: de.drugName,
                  position: 'top',
                  fontSize: 9,
                  fill: 'var(--chart-drug-marker)',
                  fontFamily: 'var(--font-mono)',
                }}
              />
            ))}

            {/* Vital value area + line */}
            <Area
              yAxisId="value"
              type="monotone"
              dataKey={valueKey}
              stroke="none"
              fill={`url(#grad-${valueKey})`}
              connectNulls
              isAnimationActive={false}
            />
            <Line
              yAxisId="value"
              type="monotone"
              dataKey={valueKey}
              stroke="#4fc3f7"
              strokeWidth={1.5}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />

            {/* Entropy overlay (dashed) */}
            <Line
              yAxisId="entropy"
              type="monotone"
              dataKey={entropyKey}
              stroke="#ba68c8"
              strokeWidth={1.2}
              strokeDasharray="4 3"
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default React.memo(VitalChart);
