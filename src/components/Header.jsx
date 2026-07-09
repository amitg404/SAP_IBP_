// Header — Multi-Agent SAP IBP branding + live status indicator
import React from 'react';
import { UserCircle, Activity, BarChart2 } from 'lucide-react';

export default function Header({ isOnline, activePersona }) {
  const getPersonaLabel = () => {
    switch (activePersona) {
      case 'blake': return { name: 'Blake (Data Analyst)', color: 'var(--c-accent-1)' };
      case 'chris': return { name: 'Chris (Forecaster)', color: 'var(--c-success)' };
      default:      return { name: 'Concierge Router', color: 'var(--c-text-muted)' };
    }
  };
  const current = getPersonaLabel();

  return (
    <header className="header">
      <div className="header-logo">
        <Activity size={20} color="#fff" />
      </div>
      <div className="header-info">
        <div className="header-title">SAP IBP Multi-Agent Hub</div>
        <div className="header-subtitle" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span>Active Context:</span>
          <span style={{ color: current.color, fontWeight: 600 }}>{current.name}</span>
        </div>
      </div>
      <div className="header-status">
        <span className={`status-dot ${isOnline ? '' : 'offline'}`} />
        {isOnline ? 'System Online' : 'System Offline'}
      </div>
    </header>
  );
}
