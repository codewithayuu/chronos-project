import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Cube, ArrowsClockwise } from '@phosphor-icons/react';
import { API_BASE } from '../utils/constants';
import { detailPanelVariants, SPRING_SNAPPY } from '../utils/animations';
import './DigitalTwinWrapper.css';

/**
 * DigitalTwinWrapper — Integration shell for 3D body model.
 *
 * Fetches data from /digital-twin/{patientId} and either:
 * 1. Renders a 3D body model (when DigitalTwinBody component is available)
 * 2. Falls back to a 2D region-based display
 *
 * The 3D component (DigitalTwinBody) will be created by another AI agent
 * and should accept these props:
 *   - regions: object with body region data (danger_level, color, status, alerts)
 *   - onRegionClick: callback(regionName)
 *   - drugMasked: boolean
 *   - activeDrugs: array
 */

function RegionCard({ name, data, onClick }) {
  if (!data) return null;
  const dangerPct = ((data.danger_level || 0) * 100).toFixed(0);

  return (
    <motion.div
      className={`twin-region-card twin-region-${data.status?.toLowerCase() || 'normal'}`}
      style={{ '--region-color': data.color || '#616161' }}
      onClick={() => onClick && onClick(name)}
      whileHover={{ scale: 1.02, borderColor: data.color }}
      transition={{ ...SPRING_SNAPPY }}
    >
      <div className="twin-region-header">
        <span className="twin-region-name">{name}</span>
        <span className="twin-region-status" style={{ color: data.color }}>
          {data.status}
        </span>
      </div>
      <div className="twin-region-bar-track">
        <motion.div
          className="twin-region-bar-fill"
          initial={{ width: 0 }}
          animate={{ width: `${dangerPct}%` }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          style={{ background: data.color }}
        />
      </div>
      <span className="twin-region-danger">{dangerPct}% danger</span>
      {data.alerts && data.alerts.length > 0 && (
        <div className="twin-region-alerts">
          {data.alerts.map((alert, idx) => (
            <span key={idx} className="twin-region-alert-text">{alert}</span>
          ))}
        </div>
      )}
    </motion.div>
  );
}

function DigitalTwinWrapper({ patientId }) {
  const [twinData, setTwinData] = useState(null);
  const [selectedRegion, setSelectedRegion] = useState(null);

  const fetchTwinData = useCallback(async () => {
    if (!patientId) return;
    try {
      const res = await fetch(`${API_BASE}/api/v1/digital-twin/${patientId}`);
      if (res.ok) setTwinData(await res.json());
    } catch (err) {
      console.error('[API] Digital twin error:', err);
    }
  }, [patientId]);

  useEffect(() => {
    fetchTwinData();
    const interval = setInterval(fetchTwinData, 10000);
    return () => clearInterval(interval);
  }, [fetchTwinData]);

  if (!twinData || twinData.calibrating) {
    return (
      <motion.div className="detail-panel twin-panel" variants={detailPanelVariants}>
        <div className="detail-panel-header">
          <h3 className="detail-panel-title">
            <Cube size={14} weight="duotone" style={{ color: 'var(--accent-teal)' }} />
            Digital Twin
          </h3>
        </div>
        <div className="twin-calibrating">
          {twinData?.calibrating ? 'Calibrating...' : 'Loading body model...'}
        </div>
      </motion.div>
    );
  }

  const regions = twinData.regions || {};
  const overall = regions.overall || {};
  const bodyRegions = ['brain', 'heart', 'lungs', 'vessels', 'autonomic', 'abdomen'];

  return (
    <motion.div
      className="detail-panel twin-panel"
      variants={detailPanelVariants}
      style={{ '--twin-overall-color': overall.color || '#616161' }}
    >
      <div className="detail-panel-header">
        <h3 className="detail-panel-title">
          <Cube size={14} weight="duotone" style={{ color: 'var(--accent-teal)' }} />
          Digital Twin
        </h3>
        <div className="twin-header-right">
          <span className="twin-overall-status" style={{ color: overall.color }}>
            {overall.status} ({overall.composite_entropy?.toFixed(3)})
          </span>
          <motion.button
            className="twin-refresh-btn"
            onClick={fetchTwinData}
            whileHover={{ scale: 1.1, rotate: 90 }}
            whileTap={{ scale: 0.9 }}
            transition={{ ...SPRING_SNAPPY }}
          >
            <ArrowsClockwise size={13} weight="bold" />
          </motion.button>
        </div>
      </div>

      {twinData.drug_masked && (
        <div className="twin-drug-warning">
          Drug masking active — some regions may appear healthier than actual state
        </div>
      )}

      {/* 2D Region Grid (3D model placeholder) */}
      <div className="twin-region-grid" id="digital-twin-mount">
        {bodyRegions.map((name) => (
          <RegionCard
            key={name}
            name={name.charAt(0).toUpperCase() + name.slice(1)}
            data={regions[name]}
            onClick={() => setSelectedRegion(name)}
          />
        ))}
      </div>

      {selectedRegion && regions[selectedRegion] && (
        <motion.div
          className="twin-region-detail"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          key={selectedRegion}
        >
          <h4>{selectedRegion.charAt(0).toUpperCase() + selectedRegion.slice(1)} — Details</h4>
          <div className="twin-metrics-grid">
            {regions[selectedRegion].metrics && Object.entries(regions[selectedRegion].metrics).map(([key, val]) => (
              <div key={key} className="twin-metric-item">
                <span className="twin-metric-label">{key.replace(/_/g, ' ')}</span>
                <span className="twin-metric-value">
                  {val != null ? (typeof val === 'number' ? val.toFixed(2) : val) : '--'}
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}

export default React.memo(DigitalTwinWrapper);
