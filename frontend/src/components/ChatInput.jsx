// ChatInput — sticky textarea + send button
// props: { value, onChange, onSend, disabled }
import { useRef } from 'react';

const SendIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 2L11 13" />
    <path d="M22 2L15 22L11 13L2 9L22 2Z" />
  </svg>
);

export default function ChatInput({ value, onChange, onSend, disabled }) {
  const textareaRef = useRef(null);

  const handleKeyDown = (e) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && value.trim()) onSend();
    }
  };

  // Auto-resize textarea as user types
  const handleChange = (e) => {
    onChange(e.target.value);
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 120) + 'px';
    }
  };

  return (
    <div className="input-area">
      <div className="input-row">
        <textarea
          id="billy-chat-input"
          ref={textareaRef}
          className="chat-input"
          rows={1}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder={disabled ? 'Billy is thinking…' : 'Ask about inventory — e.g. "Show trend for Product A in Europe"'}
          aria-label="Chat message input"
          autoComplete="off"
        />
        <button
          id="billy-send-btn"
          className="send-btn"
          onClick={onSend}
          disabled={disabled || !value.trim()}
          aria-label="Send message"
          title="Send (Enter)"
        >
          <SendIcon />
        </button>
      </div>
      <div className="input-footer">
        Inventory queries only · Phase 1 MVP · Powered by Gemma 4 via Ollama
      </div>
    </div>
  );
}
