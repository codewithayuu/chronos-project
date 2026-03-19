// =============================================
// CHART HELPER UTILITIES
// Shared formatting, color, and domain logic
// for all chart components.
// =============================================

export function formatChartTime(timestamp) {
  if (!timestamp) return '';
  const d = new Date(timestamp);
  return d.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}

export function formatChartTimeWithSeconds(timestamp) {
  if (!timestamp) return '';
  const d = new Date(timestamp);
  return d.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

export function getEntropyColor(value) {
  if (value == null) return 'var(--text-dim)';
  if (value < 0.2) return '#ff1744';
  if (value < 0.35) return '#ff6e40';
  if (value < 0.55) return '#ffd740';
  return '#00e676';
}

export function getSeverityFromCES(ces) {
  if (ces == null) return 'NONE';
  if (ces < 0.2) return 'CRITICAL';
  if (ces < 0.35) return 'WARNING';
  if (ces < 0.55) return 'WATCH';
  return 'NONE';
}

export function computeAutoDomain(data, key, thresholdLow, thresholdHigh, padding = 0.1) {
  if (!data || data.length === 0) return ['auto', 'auto'];

  const values = data.map((d) => d[key]).filter((v) => v != null);
  if (values.length === 0) return ['auto', 'auto'];

  let min = Math.min(...values);
  let max = Math.max(...values);

  if (thresholdLow != null) min = Math.min(min, thresholdLow);
  if (thresholdHigh != null) max = Math.max(max, thresholdHigh);

  const range = max - min || 5;
  const pad = range * padding;

  return [Math.floor(min - pad), Math.ceil(max + pad)];
}

export function downsampleData(data, maxPoints = 300) {
  if (!data || data.length <= maxPoints) return data;

  const step = Math.ceil(data.length / maxPoints);
  const sampled = [];
  for (let i = 0; i < data.length; i += step) {
    sampled.push(data[i]);
  }
  // Always include the last point
  if (sampled[sampled.length - 1] !== data[data.length - 1]) {
    sampled.push(data[data.length - 1]);
  }
  return sampled;
}

export const CHART_COLORS = {
  value: '#4fc3f7',
  valueGradientStart: 'rgba(79, 195, 247, 0.15)',
  valueGradientEnd: 'rgba(79, 195, 247, 0)',
  entropy: '#ba68c8',
  entropyFaded: 'rgba(186, 104, 200, 0.5)',
  threshold: '#333355',
  drugMarker: '#ffd740',
  grid: 'rgba(255, 255, 255, 0.04)',
  cesGradientStart: 'rgba(0, 230, 118, 0.12)',
  cesGradientEnd: 'rgba(0, 230, 118, 0)',
};
