import { useState, useCallback } from 'react';
import { API_BASE } from '../utils/constants';

export function usePatientHistory() {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchHistory = useCallback(async (patientId, hours = 6) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${API_BASE}/api/v1/patients/${patientId}/history?hours=${hours}` 
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setHistory(data);
    } catch (err) {
      console.error('[API] History fetch error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  return { history, loading, error, fetchHistory };
}
