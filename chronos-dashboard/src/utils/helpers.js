import { SEVERITY_ORDER } from './constants';

export function sortPatientsBySeverity(patientsObj) {
  const patientsArray = Object.values(patientsObj);
  return patientsArray.sort((a, b) => {
    const sevA = a.alert_severity || a.alert?.severity || 'NONE';
    const sevB = b.alert_severity || b.alert?.severity || 'NONE';
    const indexA = SEVERITY_ORDER.indexOf(sevA);
    const indexB = SEVERITY_ORDER.indexOf(sevB);

    if (indexA !== indexB) return indexA - indexB;

    const cesA = a.composite_entropy ?? 1;
    const cesB = b.composite_entropy ?? 1;
    return cesA - cesB;
  });
}

export function formatCES(value) {
  if (value == null) return '--';
  return value.toFixed(2);
}

export function formatVital(value) {
  if (value == null) return '--';
  return Number.isInteger(value) ? value.toString() : value.toFixed(1);
}

export function formatTime(timestamp) {
  if (!timestamp) return '--:--';
  const date = new Date(timestamp);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}

export function formatTimeAgo(timestamp) {
  if (!timestamp) return '';
  const now = new Date();
  const then = new Date(timestamp);
  const diffMs = now - then;
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 1) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  return `${diffHr}h ${diffMin % 60}m ago`;
}

export function getSeverityFromCES(ces) {
  if (ces == null) return 'NONE';
  if (ces < 0.2) return 'CRITICAL';
  if (ces < 0.35) return 'WARNING';
  if (ces < 0.55) return 'WATCH';
  return 'NONE';
}

export function getTrendSymbol(trend) {
  switch (trend) {
    case 'rising':
      return { symbol: 'rising', rotation: -45 };
    case 'falling':
      return { symbol: 'falling', rotation: 45 };
    case 'stable':
    default:
      return { symbol: 'stable', rotation: 0 };
  }
}

export function classNames(...classes) {
  return classes.filter(Boolean).join(' ');
}
