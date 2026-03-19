export const SEVERITY_ORDER = ['CRITICAL', 'WARNING', 'WATCH', 'NONE'];

export const SEVERITY_CONFIG = {
  NONE: {
    label: 'Stable',
    color: 'var(--color-none)',
    bg: 'var(--color-none-bg)',
    border: 'var(--color-none-border)',
    shadow: 'var(--shadow-glow-none)',
    animation: 'none',
    pulseClass: '',
  },
  WATCH: {
    label: 'Watch',
    color: 'var(--color-watch)',
    bg: 'var(--color-watch-bg)',
    border: 'var(--color-watch-border)',
    shadow: 'var(--shadow-glow-watch)',
    animation: 'watch-pulse 3s infinite',
    pulseClass: 'pulse-watch',
  },
  WARNING: {
    label: 'Warning',
    color: 'var(--color-warning)',
    bg: 'var(--color-warning-bg)',
    border: 'var(--color-warning-border)',
    shadow: 'var(--shadow-glow-warning)',
    animation: 'warning-pulse 2s infinite',
    pulseClass: 'pulse-warning',
  },
  CRITICAL: {
    label: 'Critical',
    color: 'var(--color-critical)',
    bg: 'var(--color-critical-bg)',
    border: 'var(--color-critical-border)',
    shadow: 'var(--shadow-glow-critical)',
    animation: 'critical-pulse 1s infinite',
    pulseClass: 'pulse-critical',
  },
};

export const VITAL_LABELS = {
  heart_rate: { label: 'Heart Rate', unit: 'bpm', icon: 'HeartPulse' },
  spo2: { label: 'SpO2', unit: '%', icon: 'Lungs' },
  bp_systolic: { label: 'BP Sys', unit: 'mmHg', icon: 'Drop' },
  bp_diastolic: { label: 'BP Dia', unit: 'mmHg', icon: 'Drop' },
  resp_rate: { label: 'Resp Rate', unit: '/min', icon: 'Wind' },
  temperature: { label: 'Temp', unit: 'C', icon: 'Thermometer' },
};

export const VITAL_THRESHOLDS = {
  heart_rate: { low: 50, high: 120 },
  spo2: { low: 90, high: null },
  bp_systolic: { low: 90, high: 180 },
  bp_diastolic: { low: 50, high: 110 },
  resp_rate: { low: 8, high: 30 },
  temperature: { low: 35.5, high: 38.5 },
};

export const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
export const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';

export const MAX_SPARKLINE_POINTS = 120;
