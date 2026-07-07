// MessageBubble — renders a single chat message (user or billy)
// props: { role: 'user'|'billy'|'error', content: string, timestamp: string, chart?: object }
import ChartWidget from './ChartWidget';

export default function MessageBubble({ role, content, timestamp, chart }) {
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
        {/* Chart rendered below the text bubble, only when present */}
        {chart && <ChartWidget chart={chart} />}
        {timestamp && (
          <div className="message-time">{timestamp}</div>
        )}
      </div>
    </div>
  );
}
