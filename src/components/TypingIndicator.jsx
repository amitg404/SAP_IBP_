// TypingIndicator — animated "Billy is analysing..." shown while awaiting response
export default function TypingIndicator() {
  return (
    <div className="typing-row">
      <div className="message-avatar billy-avatar">B</div>
      <div className="typing-bubble">
        <span className="typing-label">Billy is analysing inventory…</span>
        <div className="typing-dots">
          <div className="typing-dot" />
          <div className="typing-dot" />
          <div className="typing-dot" />
        </div>
      </div>
    </div>
  );
}
