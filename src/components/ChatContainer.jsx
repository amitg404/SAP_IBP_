// ChatContainer — scrollable message history window
// props: { messages, isLoading, onChipClick }
import React, { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';
import TypingIndicator from './TypingIndicator';
import { Network } from 'lucide-react';

const SUGGESTION_CHIPS = [
  'Show inventory trend for Product A',
  'Forecast demand for Product B',
  'What if demand drops by 10%?',
  'Compare Product C across regions',
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
          <div className="welcome-icon" aria-hidden="true"><Network color="#fff" size={32} strokeWidth={2} /></div>
          <h2>SAP IBP Multi-Agent Hub</h2>
          <p>
            Your AI-powered Supply Chain Concierge. Ask me anything about
            historical data or future demand forecasts — I'll route your request
            to the right expert in our network.
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
              persona={msg.persona}
            />
          ))}
          {isLoading && <TypingIndicator />}
        </>
      )}
      <div ref={bottomRef} aria-hidden="true" />
    </main>
  );
}
