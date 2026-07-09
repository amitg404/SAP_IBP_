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

const FALLBACK_OFFLINE =
  'Service currently unavailable. Please ensure the backend is running.';

// ── Utilities ─────────────────────────────────────────────────────────────────
let _idCounter = 0;
const uid  = () => `msg-${++_idCounter}-${Date.now()}`;
const now  = () =>
  new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

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
  const [activePersona, setActivePersona] = useState(null); // FIX 1: stateless persona

  // Session ID — generated once per tab mount
  const sessionId = useRef(crypto.randomUUID());

  // Active persona stored in ref so it is always current inside async handlers
  const activePersonaRef = useRef(null);
  useEffect(() => { activePersonaRef.current = activePersona; }, [activePersona]);

  // ── Health-check ──────────────────────────────────────────────────────────
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

  // ── Append a message to history ───────────────────────────────────────────
  const addMessage = useCallback((role, content, chart = null, persona = 'router') => {
    setMessages((prev) => [
      ...prev,
      { id: uid(), role, content, chart, persona, timestamp: now() },
    ]);
  }, []);

  // ── Core send handler ─────────────────────────────────────────────────────
  const handleSend = useCallback(async (overrideText) => {
    const text = (overrideText ?? inputValue).trim();
    if (!text || isLoading) return;

    setInput('');
    addMessage('user', text);
    setLoading(true);

    try {
      const res = await fetchWithTimeout(
        API_URL,
        {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message:        text,
            session_id:     sessionId.current,
            active_persona: activePersonaRef.current,  // FIX 1: send current persona
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

      // FIX 1: update persona state from server response
      const nextPersona = data.active_persona ?? null;
      setActivePersona(nextPersona);

      addMessage('billy', data.response ?? FALLBACK_OFFLINE, data.chart ?? null, data.persona ?? 'router');

    } catch (err) {
      if (err.name === 'AbortError') {
        addMessage('error', 'Request timed out after 90 seconds. Please try again.');
      } else {
        addMessage('error', FALLBACK_OFFLINE);
      }
    } finally {
      setLoading(false);
    }
  }, [inputValue, isLoading, addMessage]);

  const handleChipClick = useCallback((text) => {
    if (!isLoading) handleSend(text);
  }, [isLoading, handleSend]);

  return (
    <div className="app">
      <Header isOnline={isOnline} activePersona={activePersona} />
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
