// Header — Billy branding + live status indicator
export default function Header({ isOnline }) {
  return (
    <header className="header">
      <div className="header-logo">B</div>
      <div className="header-info">
        <div className="header-title">Billy</div>
        <div className="header-subtitle">SAP IBP · Inventory Assistant · Phase 1 MVP</div>
      </div>
      <div className="header-status">
        <span className={`status-dot ${isOnline ? '' : 'offline'}`} />
        {isOnline ? 'Online' : 'Offline'}
      </div>
    </header>
  );
}
