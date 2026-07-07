// MessageBubble — renders a single chat message (user or billy)
// props: { role: 'user'|'billy'|'error', content: string, timestamp: string }
export default function MessageBubble({ role, content, timestamp }) {
  const isBilly = role !== 'user';

  return (
    <div className={`message-row ${role === 'user' ? 'user' : 'billy'}`}>
      <div className={`message-avatar ${isBilly ? 'billy-avatar' : 'user-avatar'}`}>
        {isBilly ? 'B' : 'P'}
      </div>
      <div className="message-content">
        <div className={`bubble ${role === 'user' ? 'user' : role === 'error' ? 'error billy' : 'billy'}`}>
          {content}
        </div>
        {timestamp && (
          <div className="message-time">{timestamp}</div>
        )}
      </div>
    </div>
  );
}
