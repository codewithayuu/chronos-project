import React from 'react';
import { motion } from 'framer-motion';
import { Pill, Clock, Info } from '@phosphor-icons/react';
import { formatTime } from '../utils/helpers';
import { SPRING_SNAPPY } from '../utils/animations';
import './DrugBadge.css';

function DrugBadge({ drug, alert }) {
  const hasContext =
    alert?.message && drug.drug_name && alert.message.includes(drug.drug_name);

  return (
    <motion.div
      className="drug-badge"
      whileHover={{
        borderColor: 'rgba(0, 191, 165, 0.25)',
      }}
      transition={{ ...SPRING_SNAPPY }}
    >
      <div className="drug-badge-header">
        <div className="drug-badge-name-group">
          <Pill size={13} weight="duotone" className="drug-badge-icon" />
          <span className="drug-badge-name">{drug.drug_name}</span>
        </div>
        <span className="drug-badge-class">{drug.drug_class}</span>
      </div>

      <div className="drug-badge-details">
        <span className="drug-badge-dose">
          {drug.dose} {drug.unit}
        </span>
        {drug.start_time && (
          <span className="drug-badge-time">
            <Clock size={10} weight="bold" />
            {formatTime(drug.start_time)}
          </span>
        )}
      </div>

      {hasContext && (
        <motion.div
          className="drug-badge-context"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          transition={{ delay: 0.2, duration: 0.3 }}
        >
          <Info size={11} weight="duotone" />
          <span>Vital changes partially explained by this medication</span>
        </motion.div>
      )}
    </motion.div>
  );
}

export default React.memo(DrugBadge);
