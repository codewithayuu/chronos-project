import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cube, ArrowsClockwise, WarningOctagon, Heartbeat, Wind, Drop, Thermometer, Lightning } from '@phosphor-icons/react';
import { API_BASE } from '../utils/constants';
import { detailPanelVariants, SPRING_SNAPPY } from '../utils/animations';
import './DigitalTwinWrapper.css';

const REGION_ICONS = {
  Brain: Cube,
  Heart: Heartbeat,
  Lungs: Wind,
  Vessels: Drop,
  Autonomic: Lightning,
  Abdomen: Thermometer,
};

const BODY_REGIONS_LAYOUT = [
  { name: 'Brain', top: '2%', left: '50%', transform: 'translateX(-50%)' },
  { name: 'Heart', top: '28%', left: '42%', transform: 'translateX(-50%)' },
  { name: 'Lungs', top: '25%', left: '58%', transform: 'translateX(-50%)' },
  { name: 'Vessels', top: '50%', left: '28%', transform: 'translateX(-50%)' },
  { name: 'Autonomic', top: '42%', left: '50%', transform: 'translateX(-50%)' },
  { name: 'Abdomen', top: '58%', left: '58%', transform: 'translateX(-50%)' },
];

function RegionDot({ name, data, isSelected, onClick, style }) {
  if (!data) return null;
  
  const danger = data.danger_level || 0;
  const color = data.color || '#616161';
  const hasAlerts = data.alerts && data.alerts.length > 0;
  const Icon = REGION_ICONS[name] || Cube;

  return (
    <motion.button
      className={`twin-dot ${isSelected ? 'twin-dot-selected' : ''} ${danger > 0.5 ? 'twin-dot-danger' : ''}`}
      style={{
        ...style,
        '--dot-color': color,
      }}
      onClick={() => onClick(name.toLowerCase())}
      whileHover={{ scale: 1.2 }}
      whileTap={{ scale: 0.95 }}
      animate={danger > 0.7 ? {
        boxShadow: [
          `0 0 8px ${color}60`,
          `0 0 24px ${color}90`,
          `0 0 8px ${color}60`,
        ],
      } : {}}
      transition={danger > 0.7 ? { duration: 1, repeat: Infinity } : { ...SPRING_SNAPPY }}
      title={`${name}: ${data.status} (${(danger * 100).toFixed(0)}%)`}
    >
      <Icon size={14} weight="fill" />
      {hasAlerts && (
        <motion.span
          className="twin-dot-alert-pip"
          animate={{ scale: [1, 1.3, 1] }}
          transition={{ duration: 1, repeat: Infinity }}
        />
      )}
    </motion.button>
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
            Body Systems Monitor
          </h3>
        </div>
        <div className="twin-calibrating">Initializing body systems...</div>
      </motion.div>
    );
  }

  const regions = twinData.regions || {};
  const overall = regions.overall || {};
  const bodyRegions = ['brain', 'heart', 'lungs', 'vessels', 'autonomic', 'abdomen'];

  // Count danger regions
  const dangerCount = bodyRegions.filter(r => regions[r] && regions[r].danger_level > 0.5).length;
  const criticalCount = bodyRegions.filter(r => regions[r] && regions[r].danger_level > 0.75).length;

  return (
    <motion.div
      className="detail-panel twin-panel"
      variants={detailPanelVariants}
      style={{ '--twin-overall-color': overall.color || '#616161' }}
    >
      <div className="detail-panel-header">
        <h3 className="detail-panel-title">
          <Cube size={14} weight="duotone" style={{ color: 'var(--accent-teal)' }} />
          Body Systems
        </h3>
        <div className="twin-header-right">
          {criticalCount > 0 && (
            <span className="twin-critical-badge">{criticalCount} critical</span>
          )}
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

      {/* Visual Body Display */}
      <div className="twin-body-container">
        {/* Body silhouette background */}
        <div className="twin-body-silhouette">
          {/* Head */}
          <div className="twin-body-head" style={{ borderColor: regions.brain?.color || '#333' }} />
          {/* Torso */}
          <div className="twin-body-torso" style={{ borderColor: overall.color || '#333' }} />
          {/* Arms */}
          <div className="twin-body-arm twin-body-arm-left" style={{ borderColor: regions.vessels?.color || '#333' }} />
          <div className="twin-body-arm twin-body-arm-right" style={{ borderColor: regions.vessels?.color || '#333' }} />
          {/* Legs */}
          <div className="twin-body-leg twin-body-leg-left" style={{ borderColor: regions.vessels?.color || '#333' }} />
          <div className="twin-body-leg twin-body-leg-right" style={{ borderColor: regions.vessels?.color || '#333' }} />
          
          {/* Scan line effect */}
          <motion.div
            className="twin-scan-line"
            animate={{ top: ['0%', '100%', '0%'] }}
            transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
          />
        </div>

        {/* Region dots positioned on body */}
        {BODY_REGIONS_LAYOUT.map((layout) => (
          <RegionDot
            key={layout.name}
            name={layout.name}
            data={regions[layout.name.toLowerCase()]}
            isSelected={selectedRegion === layout.name.toLowerCase()}
            onClick={setSelectedRegion}
            style={{
              position: 'absolute',
              top: layout.top,
              left: layout.left,
              transform: layout.transform,
            }}
          />
        ))}
      </div>

      {/* Overall status bar */}
      <div className="twin-overall-bar">
        <span className="twin-overall-label">Overall Status</span>
        <div className="twin-overall-track">
          <motion.div
            className="twin-overall-fill"
            animate={{ width: `${((overall.composite_entropy || 0) * 100).toFixed(0)}%` }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            style={{ background: overall.color || '#616161' }}
          />
        </div>
        <span className="twin-overall-value" style={{ color: overall.color }}>{overall.status}</span>
      </div>

      {/* Region status list */}
      <div className="twin-region-list">
        {bodyRegions.map((name) => {
          const data = regions[name];
          if (!data) return null;
          const danger = data.danger_level || 0;
          const isExpanded = selectedRegion === name;
          
          return (
            <motion.div
              key={name}
              className={`twin-region-item ${danger > 0.75 ? 'twin-region-critical' : danger > 0.5 ? 'twin-region-warning' : ''}`}
              onClick={() => setSelectedRegion(isExpanded ? null : name)}
              style={{ cursor: 'pointer' }}
            >
              <div className="twin-region-item-header">
                <div className="twin-region-item-left">
                  <span className="twin-region-dot-indicator" style={{ background: data.color }} />
                  <span className="twin-region-item-name">{name}</span>
                </div>
                <div className="twin-region-item-right">
                  <span className="twin-region-item-pct" style={{ color: data.color }}>
                    {(danger * 100).toFixed(0)}%
                  </span>
                  <span className="twin-region-item-status" style={{ color: data.color }}>
                    {data.status}
                  </span>
                </div>
              </div>

              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    className="twin-region-expanded"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    {data.alerts && data.alerts.length > 0 && (
                      <div className="twin-region-alerts-list">
                        {data.alerts.map((alert, idx) => (
                          <div key={idx} className="twin-region-alert-item">
                            <WarningOctagon size={10} weight="fill" style={{ color: data.color }} />
                            <span>{alert}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {data.metrics && (
                      <div className="twin-region-metrics">
                        {Object.entries(data.metrics).map(([key, val]) => (
                          val != null && (
                            <div key={key} className="twin-region-metric">
                              <span className="twin-region-metric-key">{key.replace(/_/g, ' ')}</span>
                              <span className="twin-region-metric-val">
                                {typeof val === 'number' ? val.toFixed(2) : String(val)}
                              </span>
                            </div>
                          )
                        ))}
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>

      {twinData.drug_masked && (
        <div className="twin-drug-warning">
          <WarningOctagon size={12} weight="fill" />
          <span>Drug masking — some regions may appear healthier than actual</span>
        </div>
      )}
    </motion.div>
  );
}

export default React.memo(DigitalTwinWrapper);
