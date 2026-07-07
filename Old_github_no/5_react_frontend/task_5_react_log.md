# Task 5 — React Chat Interface Log

**Billy MVP — React Frontend**  
Framework: Vite 8 + React 19 | Bundler: Vite | Target: Vercel

---

## 1. Component Hierarchy

```
App (src/App.jsx)
 │  Owns: messages[], inputValue, isLoading, isOnline
 │  Handles: API fetch, timeout, error catching, health polling
 │
 ├── Header (src/components/Header.jsx)
 │    Props: isOnline (bool)
 │    Renders: Billy logo, subtitle, live status dot (green/red)
 │
 ├── ChatContainer (src/components/ChatContainer.jsx)
 │    Props: messages[], isLoading, onChipClick
 │    Renders: welcome screen (empty state) OR message list + typing indicator
 │    Auto-scrolls to latest message via useEffect + ref
 │
 │    ├── MessageBubble (src/components/MessageBubble.jsx)
 │    │    Props: role ('user'|'billy'|'error'), content, timestamp
 │    │    Variants: blue gradient (user), dark glass (billy), red tint (error)
 │    │
 │    └── TypingIndicator (src/components/TypingIndicator.jsx)
 │         No props — shown only when isLoading=true
 │         Animated 3-dot bounce with label "Billy is analysing inventory…"
 │
 └── ChatInput (src/components/ChatInput.jsx)
      Props: value, onChange, onSend, disabled
      Auto-resizes textarea up to 120px
      Enter to send, Shift+Enter for new line
      Disabled + placeholder changes during loading
```

---

## 2. Environment Configuration

### Local Development (`.env`)
```bash
VITE_API_URL=http://localhost:8000
```

### Production (`.env.production`)
```bash
VITE_API_URL=https://your-billy-backend.vercel.app
```

**How it's used in code:**
```js
const API_URL    = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000') + '/chat';
const HEALTH_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000') + '/health';
```

The `??` fallback means the app works even with no `.env` file present.

> **Vercel:** Set `VITE_API_URL` as an **Environment Variable** in Project Settings → Environment Variables → Production. Vite bakes it into the bundle at build time.

---

## 3. Running Locally

```bash
# From 5_react_frontend directory:

# Install dependencies (first time only)
npm install

# Start dev server (hot reload)
npm run dev
# → App available at http://localhost:5173

# Production build (verify before deploy)
npm run build

# Preview production build locally
npm run preview
```

**Also start the FastAPI backend** in a separate terminal:
```bash
# From 4_fastapi_backend directory:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 4. Error Handling Architecture

```
User submits query
       │
       ├─ [disabled check]  isLoading=true → button disabled, no re-submit
       │
       ├─ fetchWithTimeout (60s AbortController)
       │       │
       │       ├─ AbortError (timeout) → error bubble "Request timed out…"
       │       ├─ Network error        → error bubble "Service unavailable…"
       │       ├─ res.ok=false (4xx/5xx) → error bubble from res.json().response
       │       └─ JSON parse failure   → error bubble "Service unavailable…"
       │
       └─ finally: setLoading(false) → input re-enabled
```

**Health polling:** `GET /health` runs every 15 s. Status dot turns red if unreachable — visual warning before the user even sends a message.

---

## 5. Key UX Decisions

| Decision | Reason |
|---|---|
| 60 s `AbortController` timeout | Local LLM cold start can be 30–45 s; 60 s prevents premature failure |
| Input disabled during request | Prevents spamming Ollama, which would queue and slow responses |
| Chip buttons on welcome screen | Demo-ready shortcuts — click once to trigger the hero demo question |
| `error` role bubble (red tint) | Visually distinct from Billy's answer — planners know it's a system issue |
| Health polling every 15 s | Status dot turns red if Ollama/FastAPI drops mid-demo |
| `aria-live="polite"` on chat window | Screen-reader accessibility |

---

## 6. Deploying to Vercel

### One-time setup

```bash
# Install Vercel CLI (global)
npm install -g vercel

# Login
vercel login
```

### Deploy from the `5_react_frontend` subfolder

```bash
# From 5_react_frontend directory:
vercel

# Follow prompts:
#  ? Set up and deploy? → Y
#  ? Which scope? → (your team/account)
#  ? Link to existing project? → N (first deploy)
#  ? Project name → billy-frontend
#  ? Directory → ./  (already in 5_react_frontend)
#  ? Override build settings? → N  (Vite auto-detected)
```

### Set environment variable on Vercel

```bash
vercel env add VITE_API_URL production
# Paste: https://your-billy-fastapi-backend.railway.app  (or wherever FastAPI is hosted)
```

### Subsequent deploys

```bash
vercel --prod
```

### `vercel.json` (already included)
```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/" }]
}
```
Ensures React Router-style navigation doesn't 404 on hard refresh.

---

## 7. Build Verification

```
vite v8.1.3 — production build
  dist/index.html          0.88 kB  │ gzip:  0.54 kB
  dist/assets/index.css    7.82 kB  │ gzip:  2.35 kB
  dist/assets/index.js   195.92 kB  │ gzip: 62.08 kB
  Built in 202ms — 0 errors, 0 warnings
```

---

## 8. Architecture Connection

```
Planner opens http://localhost:5173
       │
       └─► React App (Vite)
               │  POST /chat {"message": "..."}
               ▼
       FastAPI backend :8000  (Task 4)
               │  graph.invoke()
               ▼
       LangGraph agent (Task 2)
               │  get_inventory tool (Task 2) → inventory.csv (Task 1)
               │  system prompt (Task 3)
               ▼
       Ollama / Gemma 4 12B (local GPU)
               │
               └─► {"response": "Inventory for Product A has risen 32.4%..."}
               │
               ▼
       React renders conversational bubble
       ← Answer in <30 seconds, no SAP login, no Excel ←
```

---

*Generated: 2026-07-07 | Billy MVP — Task 5 Complete*
