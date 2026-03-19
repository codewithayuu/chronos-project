import { useEffect, useRef, useCallback } from 'react';
import { API_BASE } from '../utils/constants';

/**
 * VoiceAlertSystem — speaks critical alerts using browser TTS.
 *
 * Fetches voice-formatted text from backend endpoint,
 * then uses browser's SpeechSynthesis API to speak it.
 *
 * Props:
 *   alerts: array of alert objects from WebSocket
 *   enabled: boolean to enable/disable voice
 */
function VoiceAlertSystem({ alerts, enabled = true }) {
  const spokenAlerts = useRef(new Set());
  const lastAlertCount = useRef(0);

  const speakAlert = useCallback(async (patientId) => {
    if (!('speechSynthesis' in window)) return;

    try {
      const res = await fetch(
        `${API_BASE}/api/v1/voice-alert/${patientId}` 
      );
      if (!res.ok) return;
      const data = await res.json();

      const text = data.voice_text;
      if (!text) return;

      // Cancel any current speech
      window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = data.rate || 0.9;
      utterance.pitch = data.pitch || 0.8;
      utterance.volume = 1.0;
      utterance.lang = 'en-US';

      // Try to find a good voice
      const voices = window.speechSynthesis.getVoices();
      const preferred = voices.find(
        (v) =>
          v.lang.startsWith('en') &&
          (v.name.includes('Google') ||
            v.name.includes('Microsoft') ||
            v.name.includes('Samantha') ||
            v.name.includes('Daniel'))
      );
      if (preferred) {
        utterance.voice = preferred;
      }

      window.speechSynthesis.speak(utterance);
    } catch (err) {
      console.error('[Voice] Error:', err);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    if (!alerts || alerts.length === 0) return;
    if (alerts.length <= lastAlertCount.current) {
      lastAlertCount.current = alerts.length;
      return;
    }
    lastAlertCount.current = alerts.length;

    // Check newest alert (first in array)
    const newest = alerts[0];
    if (!newest) return;

    const alertKey = `${newest.patient_id}-${newest.timestamp}`;
    if (spokenAlerts.current.has(alertKey)) return;

    // Only speak WARNING and CRITICAL
    if (newest.severity === 'CRITICAL' || newest.severity === 'WARNING') {
      spokenAlerts.current.add(alertKey);
      speakAlert(newest.patient_id);

      // Clean up old entries
      if (spokenAlerts.current.size > 50) {
        const entries = Array.from(spokenAlerts.current);
        entries.slice(0, 25).forEach((e) => spokenAlerts.current.delete(e));
      }
    }
  }, [alerts, enabled, speakAlert]);

  // Load voices on mount
  useEffect(() => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.getVoices();
      window.speechSynthesis.onvoiceschanged = () => {
        window.speechSynthesis.getVoices();
      };
    }
  }, []);

  return null;
}

export default VoiceAlertSystem;
