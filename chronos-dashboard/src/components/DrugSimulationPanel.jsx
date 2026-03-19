import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Syringe,
  MagnifyingGlass,
  Play,
  Check,
  Warning,
  ArrowDown,
  ArrowUp,
  Minus,
  Pill,
  Clock,
  Lightning,
  X,
} from '@phosphor-icons/react';
import AnimatedNumber from './AnimatedNumber';
import { API_BASE } from '../utils/constants';
import { detailPanelVariants, SPRING_SNAPPY } from '../utils/animations';
import './DrugSimulationPanel.css';

function EffectRow({ effect }) {
  const isIncrease = effect.direction === 'increase';
  const isDecrease = effect.direction === 'decrease';
  const Icon = isIncrease ? ArrowUp : isDecrease ? ArrowDown : Minus;
  const color = isDecrease ? 'var(--color-watch)' : isIncrease ? 'var(--color-warning)' : 'var(--text-dim)';

  return (
    <div className="drug-effect-row">
      <div className="drug-effect-vital">
        <Icon size={12} weight="bold" style={{ color }} />
        <span>{effect.vital_sign}</span>
      </div>
      <div className="drug-effect-values">
        <span className="drug-effect-current">{effect.current_value ?? '--'}</span>
        <span className="drug-effect-arrow">→</span>
        <span className="drug-effect-predicted" style={{ color }}>
          {effect.predicted_value ?? '--'}
        </span>
        <span className="drug-effect-unit">{effect.unit}</span>
      </div>
      <div className="drug-effect-change" style={{ color }}>
        {effect.predicted_change > 0 ? '+' : ''}{effect.predicted_change} {effect.unit}
      </div>
      {effect.predicted_entropy != null && (
        <div className="drug-effect-entropy">
          <span className="drug-effect-entropy-label">Entropy:</span>
          <span>{effect.current_entropy?.toFixed(2) ?? '--'}</span>
          <span>→</span>
          <span style={{ color: effect.predicted_entropy < 0.3 ? 'var(--color-critical)' : 'var(--text-secondary)' }}>
            {effect.predicted_entropy?.toFixed(2) ?? '--'}
          </span>
        </div>
      )}
    </div>
  );
}

function DrugSimulationPanel({ patientId, patient }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [drugList, setDrugList] = useState([]);
  const [filteredDrugs, setFilteredDrugs] = useState([]);
  const [selectedDrug, setSelectedDrug] = useState(null);
  const [dose, setDose] = useState('');
  const [unit, setUnit] = useState('mg');
  const [duration, setDuration] = useState('');
  const [simulation, setSimulation] = useState(null);
  const [simulating, setSimulating] = useState(false);
  const [injected, setInjected] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);

  // Load all drugs on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/v1/drugs/list`)
      .then(res => res.json())
      .then(data => {
        setDrugList(data.drugs || []);
        setFilteredDrugs(data.drugs || []);
      })
      .catch(err => console.error('[API] Drug list error:', err));
  }, []);

  // Filter drugs on search
  useEffect(() => {
    if (!searchQuery.trim()) {
      setFilteredDrugs(drugList);
      return;
    }
    const q = searchQuery.toLowerCase();
    setFilteredDrugs(
      drugList.filter(d =>
        d.drug_name?.toLowerCase().includes(q) ||
        d.drug_class?.toLowerCase().includes(q)
      )
    );
  }, [searchQuery, drugList]);

  const handleSelectDrug = (drug) => {
    setSelectedDrug(drug);
    setSearchQuery(drug.drug_name);
    setShowDropdown(false);
    setSimulation(null);
    setInjected(false);
    // Set default values
    setDose('10');
    setUnit(drug.drug_class === 'vasopressor' ? 'mcg/kg/min' : 'mg');
    setDuration(String(drug.duration_minutes || 360));
  };

  const handleSimulate = async () => {
    if (!selectedDrug || !patientId) return;
    setSimulating(true);
    setSimulation(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/drugs/simulate/${patientId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          drug_name: selectedDrug.drug_name,
          drug_class: selectedDrug.drug_class,
          dose: parseFloat(dose) || null,
          unit: unit,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setSimulation(data);
      }
    } catch (err) {
      console.error('[API] Simulation error:', err);
    } finally {
      setSimulating(false);
    }
  };

  const handleInject = async () => {
    if (!selectedDrug || !patientId) return;
    try {
      const res = await fetch(`${API_BASE}/api/v1/patients/${patientId}/drugs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          drug_name: selectedDrug.drug_name,
          drug_class: selectedDrug.drug_class,
          dose: parseFloat(dose) || null,
          unit: unit,
        }),
      });
      if (res.ok) {
        setInjected(true);
      }
    } catch (err) {
      console.error('[API] Injection error:', err);
    }
  };

  const handleReset = () => {
    setSelectedDrug(null);
    setSearchQuery('');
    setSimulation(null);
    setInjected(false);
    setDose('');
    setDuration('');
  };

  return (
    <motion.div
      className="detail-panel drug-sim-panel"
      variants={detailPanelVariants}
    >
      <div className="detail-panel-header">
        <h3 className="detail-panel-title">
          <Syringe size={14} weight="duotone" style={{ color: 'var(--accent-teal)' }} />
          Drug Simulation Lab
        </h3>
        {selectedDrug && (
          <motion.button
            className="drug-sim-reset"
            onClick={handleReset}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <X size={12} weight="bold" />
          </motion.button>
        )}
      </div>

      {/* Drug Search */}
      <div className="drug-search-container">
        <div className="drug-search-input-wrap">
          <MagnifyingGlass size={14} weight="bold" className="drug-search-icon" />
          <input
            className="drug-search-input"
            type="text"
            placeholder="Search drug name..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setShowDropdown(true);
              setSelectedDrug(null);
              setSimulation(null);
            }}
            onFocus={() => setShowDropdown(true)}
          />
        </div>

        <AnimatePresence>
          {showDropdown && filteredDrugs.length > 0 && !selectedDrug && (
            <motion.div
              className="drug-dropdown"
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
            >
              {filteredDrugs.slice(0, 8).map((drug) => (
                <button
                  key={drug.drug_name}
                  className="drug-dropdown-item"
                  onClick={() => handleSelectDrug(drug)}
                >
                  <Pill size={12} weight="duotone" />
                  <span className="drug-dropdown-name">{drug.drug_name}</span>
                  <span className="drug-dropdown-class">{drug.drug_class}</span>
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Selected Drug Details */}
      {selectedDrug && (
        <motion.div
          className="drug-selected"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="drug-selected-header">
            <div className="drug-selected-name-group">
              <Pill size={16} weight="duotone" style={{ color: 'var(--accent-teal)' }} />
              <div>
                <span className="drug-selected-name">{selectedDrug.drug_name}</span>
                <span className="drug-selected-class">{selectedDrug.drug_class}</span>
              </div>
            </div>
            <div className="drug-selected-timing">
              <Clock size={11} weight="bold" />
              <span>Onset: {selectedDrug.onset_minutes || '?'}min</span>
              <span>Duration: {selectedDrug.duration_minutes || '?'}min</span>
            </div>
          </div>

          {/* Dose Controls */}
          <div className="drug-dose-row">
            <div className="drug-dose-field">
              <label>Dose</label>
              <input
                type="number"
                className="drug-dose-input"
                value={dose}
                onChange={(e) => setDose(e.target.value)}
                placeholder="10"
              />
            </div>
            <div className="drug-dose-field">
              <label>Unit</label>
              <select className="drug-dose-select" value={unit} onChange={(e) => setUnit(e.target.value)}>
                <option value="mg">mg</option>
                <option value="mcg">mcg</option>
                <option value="mcg/kg/min">mcg/kg/min</option>
                <option value="mg/hr">mg/hr</option>
                <option value="mL">mL</option>
                <option value="units">units</option>
              </select>
            </div>
            <div className="drug-dose-field">
              <label>Duration (min)</label>
              <input
                type="number"
                className="drug-dose-input"
                value={duration}
                onChange={(e) => setDuration(e.target.value)}
                placeholder="360"
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="drug-actions">
            <motion.button
              className="drug-simulate-btn"
              onClick={handleSimulate}
              disabled={simulating}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
            >
              <Play size={14} weight="fill" />
              {simulating ? 'Simulating...' : 'SIMULATE'}
            </motion.button>

            {simulation && !injected && (
              <motion.button
                className="drug-inject-btn"
                onClick={handleInject}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
              >
                <Syringe size={14} weight="fill" />
                ADMINISTER
              </motion.button>
            )}

            {injected && (
              <motion.div
                className="drug-injected-badge"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
              >
                <Check size={14} weight="bold" />
                <span>Administered</span>
              </motion.div>
            )}
          </div>
        </motion.div>
      )}

      {/* Simulation Results */}
      <AnimatePresence>
        {simulation && (
          <motion.div
            className="drug-simulation-results"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <div className="drug-sim-header">
              <Lightning size={14} weight="duotone" style={{ color: 'var(--accent-teal)' }} />
              <span>Predicted Effects</span>
              <span className="drug-sim-entropy-impact">
                Entropy: {simulation.entropy_impact === 'reduces' ? '↓ Reduces' : simulation.entropy_impact === 'increases' ? '↑ Increases' : '→ No change'}
              </span>
            </div>

            <div className="drug-sim-ces">
              <span>Current CES:</span>
              <AnimatedNumber value={simulation.current_patient_ces} decimals={3} className="drug-sim-ces-value" />
              <span className="drug-sim-severity">{simulation.current_severity}</span>
            </div>

            {simulation.predicted_effects && simulation.predicted_effects.length > 0 ? (
              <div className="drug-effects-list">
                {simulation.predicted_effects.map((effect, idx) => (
                  <EffectRow key={idx} effect={effect} />
                ))}
              </div>
            ) : (
              <p className="drug-no-effects">No significant vital sign effects predicted for this drug.</p>
            )}

            {simulation.historical_effectiveness && (
              <div className="drug-effectiveness">
                <span className="drug-effectiveness-label">Historical Effectiveness:</span>
                <span className="drug-effectiveness-rate">
                  {(simulation.historical_effectiveness.historical_success_rate * 100).toFixed(0)}%
                </span>
                <span className="drug-effectiveness-cases">
                  (n={simulation.historical_effectiveness.similar_cases} similar cases)
                </span>
              </div>
            )}

            {simulation.warnings && simulation.warnings.length > 0 && (
              <div className="drug-warnings">
                {simulation.warnings.map((w, idx) => (
                  <div key={idx} className="drug-warning-item">
                    <Warning size={12} weight="fill" />
                    <span>{w}</span>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export default React.memo(DrugSimulationPanel);
