import React from 'react';
import { WifiHigh, WifiSlash } from '@phosphor-icons/react';

function ConnectionStatus({ connected }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '6px 12px',
        borderRadius: 'var(--radius-full)',
        fontSize: 12,
        fontWeight: 600,
        background: connected
          ? 'rgba(0, 230, 118, 0.08)'
          : 'rgba(255, 23, 68, 0.08)',
        color: connected ? 'var(--color-none)' : 'var(--color-critical)',
        border: `1px solid ${
          connected ? 'rgba(0, 230, 118, 0.15)' : 'rgba(255, 23, 68, 0.15)'
        }`,
      }}
    >
      {connected ? <WifiHigh size={14} weight="bold" /> : <WifiSlash size={14} weight="bold" />}
      <span>{connected ? 'Live' : 'Offline'}</span>
    </div>
  );
}

export default React.memo(ConnectionStatus);
