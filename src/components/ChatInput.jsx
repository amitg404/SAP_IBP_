// ChatInput — sticky textarea + send button
import React, { useRef } from 'react';
import { Send, CornerDownLeft } from 'lucide-react';

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
      <div className="input-wrapper">
        <textarea
          id="billy-chat-input"
          ref={textareaRef}
          className="chat-input"
          rows={1}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder={disabled ? 'Processing request...' : 'Ask Blake for data, or Chris for forecasts...'}
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
          <Send size={18} strokeWidth={2.5} />
        </button>
      </div>
      <div className="input-footer">
        SAP IBP Multi-Agent Hub <CornerDownLeft size={10} style={{ margin: '0 4px', display: 'inline' }} /> Powered by Multi-LLM Architecture
      </div>
    </div>
  );
}
