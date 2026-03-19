import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { useChronosWebSocket } from './hooks/useChronosWebSocket';
import SkipToContent from './components/SkipToContent';
import Header from './components/Header';
import WardView from './components/WardView';
import PatientDetailView from './components/PatientDetailView';
import AlertFeed from './components/AlertFeed';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import SplitScreenView from './components/SplitScreenView';
import VoiceAlertSystem from './components/VoiceAlertSystem';
import DangerPopup from './components/DangerPopup';
import InitialLoader from './components/InitialLoader';
import './App.css';

function AnimatedRoutes({
  patients,
  sparklines,
  alerts,
  acknowledgeAlert,
  alertFeedOpen,
  setAlertFeedOpen,
  connected,
  systemStatus,
}) {
  const location = useLocation();

  return (
    <>
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route
            path="/"
            element={
              <WardView
                patients={patients}
                sparklines={sparklines}
                connected={connected}
                systemStatus={systemStatus}
              />
            }
          />
          <Route
            path="/patient/:patientId"
            element={
              <PatientDetailView
                patients={patients}
                sparklines={sparklines}
              />
            }
          />
          <Route
            path="/analytics"
            element={<AnalyticsDashboard />}
          />
          <Route
            path="/patient/:patientId/compare"
            element={<SplitScreenView patients={patients} />}
          />
        </Routes>
      </AnimatePresence>

      <AnimatePresence>
        {alertFeedOpen && (
          <AlertFeed
            alerts={alerts}
            isOpen={alertFeedOpen}
            onClose={() => setAlertFeedOpen(false)}
            onAcknowledge={acknowledgeAlert}
          />
        )}
      </AnimatePresence>
    </>
  );
}

function App() {
  const { patients, alerts, connected, systemStatus, sparklines, acknowledgeAlert } =
    useChronosWebSocket();

  const [alertFeedOpen, setAlertFeedOpen] = React.useState(false);
  const [voiceMuted, setVoiceMuted] = React.useState(false);
  const [dangerAlert, setDangerAlert] = React.useState(null);
  const lastDangerRef = React.useRef(null);

  const activeAlertCount = alerts.filter((a) => !a.acknowledged).length;
  const hasPatients = Object.keys(patients).length > 0;

  // Close alert feed on Escape
  React.useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && alertFeedOpen) {
        setAlertFeedOpen(false);
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [alertFeedOpen]);

  // Auto-show danger popup on new WARNING/CRITICAL alerts
  React.useEffect(() => {
    if (alerts.length === 0) return;
    const newest = alerts[0];
    if (!newest || !newest.patient_id) return;
    
    const alertKey = `${newest.patient_id}-${newest.severity}-${newest.timestamp}`;
    if (alertKey === lastDangerRef.current) return;
    
    if (newest.severity === 'CRITICAL' || newest.severity === 'WARNING') {
      lastDangerRef.current = alertKey;
      setDangerAlert(newest);
    }
  }, [alerts]);

  return (
    <Router>
      <div className="app-shell" role="application" aria-label="Project Chronos ICU Monitoring Dashboard">
        <SkipToContent />

        <Header
          connected={connected}
          patientCount={Object.keys(patients).length}
          activeAlertCount={activeAlertCount}
          systemStatus={systemStatus}
          onAlertClick={() => setAlertFeedOpen(true)}
          patients={patients}
          voiceMuted={voiceMuted}
          onVoiceToggle={() => setVoiceMuted(prev => !prev)}
        />

        <VoiceAlertSystem alerts={alerts} enabled={!voiceMuted} />

        <AnimatePresence>
          {dangerAlert && (
            <DangerPopup
              alert={dangerAlert}
              patient={patients[dangerAlert.patient_id]}
              onDismiss={() => setDangerAlert(null)}
            />
          )}
        </AnimatePresence>

        <main className="app-main" id="main-content" role="main">
          {!hasPatients && connected ? (
            <InitialLoader />
          ) : (
            <AnimatedRoutes
              patients={patients}
              sparklines={sparklines}
              alerts={alerts}
              acknowledgeAlert={acknowledgeAlert}
              alertFeedOpen={alertFeedOpen}
              setAlertFeedOpen={setAlertFeedOpen}
              connected={connected}
              systemStatus={systemStatus}
            />
          )}
        </main>
      </div>
    </Router>
  );
}

export default App;
