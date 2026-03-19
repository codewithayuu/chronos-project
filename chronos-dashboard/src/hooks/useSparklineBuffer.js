import { useRef, useCallback, useMemo } from 'react';
import { MAX_SPARKLINE_POINTS } from '../utils/constants';

export function useSparklineBuffer(maxPoints = MAX_SPARKLINE_POINTS) {
  const buffers = useRef({});

  const pushValue = useCallback(
    (patientId, value, timestamp) => {
      if (value == null) return;
      if (!buffers.current[patientId]) {
        buffers.current[patientId] = [];
      }
      const buf = buffers.current[patientId];
      buf.push({ value, timestamp });
      if (buf.length > maxPoints) {
        buf.shift();
      }
    },
    [maxPoints]
  );

  const getValues = useCallback((patientId) => {
    const buf = buffers.current[patientId];
    if (!buf) return [];
    return buf.map((entry) => entry.value);
  }, []);

  const getEntries = useCallback((patientId) => {
    return buffers.current[patientId] || [];
  }, []);

  const getLatest = useCallback((patientId, count = 60) => {
    const buf = buffers.current[patientId];
    if (!buf) return [];
    return buf.slice(-count);
  }, []);

  return { pushValue, getValues, getEntries, getLatest };
}
