// ChatContainer — scrollable message history window
// props: { messages, isLoading, onChipClick }
import { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';
import TypingIndicator from './TypingIndicator';

const SUGGESTION_CHIPS = [
  'Show inventory trend for Product A',
  'Is Product B running low in Europe?',
  'Is Product C inventory stable?',
  'Compare Product A across regions',
];

export default function ChatContainer({ messages, isLoading, onChipClick }) {
  const bottomRef = useRef(null);

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const isEmpty = messages.length === 0;

  return (
    <main className="chat-container" id="chat-window" role="log" aria-live="polite">
      {isEmpty && !isLoading ? (
        <div className="welcome">
          <div className="welcome-icon" aria-hidden="true">📦</div>
          <h2>Hello, I'm Billy</h2>
          <p>
            Your AI-powered SAP IBP Inventory Assistant. Ask me anything about
            stock levels or inventory trends — I'll get you an answer in seconds,
            without opening SAP or Excel.
          </p>
          <div className="welcome-chips" role="list" aria-label="Suggested questions">
            {SUGGESTION_CHIPS.map((chip) => (
              <button
                key={chip}
                className="chip"
                role="listitem"
                onClick={() => onChipClick(chip)}
                aria-label={`Ask: ${chip}`}
              >
                {chip}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <>
          {messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              role={msg.role}
              content={msg.content}
              chart={msg.chart}
              timestamp={msg.timestamp}
            />
          ))}
          {isLoading && <TypingIndicator />}
        </>
      )}
      <div ref={bottomRef} aria-hidden="true" />
    </main>
  );
}
