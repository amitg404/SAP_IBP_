// App — root component, orchestrates all state and API communication
import { useState, useCallback, useEffect, useRef } from 'react';
import Header from './components/Header';
import ChatContainer from './components/ChatContainer';
import ChatInput from './components/ChatInput';

// ── Config ────────────────────────────────────────────────────────────────────
const API_URL    = (import.meta.env.VITE_API_URL ?? '/api') + '/chat';
const HEALTH_URL = (import.meta.env.VITE_API_URL ?? '/api') + '/health';

// Request timeout — 90 s to allow cloud model inference
const TIMEOUT_MS = 90_000;

// Fallback message shown in the chat when backend is unreachable
const FALLBACK_OFFLINE =
  'Service currently unavailable. Please ensure the local backend (FastAPI on port 8000) and Ollama are running.';

// ── Utilities ─────────────────────────────────────────────────────────────────
let _idCounter = 0;
const uid  = () => `msg-${++_idCounter}-${Date.now()}`;
const now  = () =>
  new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

// AbortController-based fetch with timeout
async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages]   = useState([]);
  const [inputValue, setInput]    = useState('');
  const [isLoading, setLoading]   = useState(false);
  const [isOnline, setOnline]     = useState(true);

  // Session ID — generated once per tab mount, persists for entire session
  const sessionId = useRef(crypto.randomUUID());

  // ── Health-check: poll backend every 15 s for live status dot ──────────────
  useEffect(() => {
    let mounted = true;

    const checkHealth = async () => {
      try {
        const res = await fetchWithTimeout(HEALTH_URL, { method: 'GET' }, 5_000);
        if (mounted) setOnline(res.ok);
      } catch {
        if (mounted) setOnline(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 15_000);
    return () => { mounted = false; clearInterval(interval); };
  }, []);

  // ── Append a message to the history ────────────────────────────────────────
  const addMessage = useCallback((role, content, chart = null) => {
    setMessages((prev) => [
      ...prev,
      { id: uid(), role, content, chart, timestamp: now() },
    ]);
  }, []);

  // ── Core send handler ───────────────────────────────────────────────────────
  const handleSend = useCallback(async (overrideText) => {
    const text = (overrideText ?? inputValue).trim();
    if (!text || isLoading) return;

    // 1. Show user message immediately, clear input
    setInput('');
    addMessage('user', text);
    setLoading(true);

    // 2. Call FastAPI backend
    try {
      const res = await fetchWithTimeout(
        API_URL,
        {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({
            message:    text,
            session_id: sessionId.current,  // conversation memory
          }),
        },
        TIMEOUT_MS,
      );

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        addMessage('error', errBody?.response ?? FALLBACK_OFFLINE);
        return;
      }

      const data = await res.json();
      // Pass chart field (null if absent) alongside text response
      addMessage('billy', data.response ?? FALLBACK_OFFLINE, data.chart ?? null);

    } catch (err) {
      if (err.name === 'AbortError') {
        addMessage('error', 'Request timed out after 90 seconds. The model may still be loading — please try again.');
      } else {
        addMessage('error', FALLBACK_OFFLINE);
      }
    } finally {
      setLoading(false);
    }
  }, [inputValue, isLoading, addMessage]);

  // ── Chip click → pre-fill and auto-send ────────────────────────────────────
  const handleChipClick = useCallback((text) => {
    if (!isLoading) handleSend(text);
  }, [isLoading, handleSend]);

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="app">
      <Header isOnline={isOnline} />
      <ChatContainer
        messages={messages}
        isLoading={isLoading}
        onChipClick={handleChipClick}
      />
      <ChatInput
        value={inputValue}
        onChange={setInput}
        onSend={handleSend}
        disabled={isLoading}
      />
    </div>
  );
}
