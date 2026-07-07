# RUNBOOK — Billy MVP Live Demo Guide

> **Audience:** Demo operator / Solutions Architect  
> **Time to stand up:** ~3 minutes  
> **Last verified:** 2026-07-07

---

## Prerequisites Checklist

Before starting, confirm all of the following:

- [ ] NVIDIA RTX 4060 (8 GB VRAM) available and drivers current
- [ ] Ollama installed → [https://ollama.com](https://ollama.com)
- [ ] Model pulled: `ollama pull gemma4:12b` *(one-time, ~7 GB download)*
- [ ] Python 3.10+ with backend dependencies installed
- [ ] Node 18+ installed
- [ ] Ports **11434**, **8000**, and **5173** are free

---

## Step 1 — Start the LLM (Ollama)

Open **Terminal 1**:

```bash
ollama serve
```

> Ollama starts on `http://localhost:11434`. Keep this terminal open.

**Verify the model is available:**
```bash
ollama list
# Should show: gemma4:12b
```

**Alternative models** (if gemma4:12b is unavailable):
```bash
ollama pull gemma4:e4b   # lighter, ~4B effective params
ollama pull qwen3:8b     # strong tool-calling alternative
```

Set the model via env var before starting the backend:
```bash
set BILLY_MODEL=qwen3:8b     # Windows
# export BILLY_MODEL=qwen3:8b  # Mac/Linux
```

---

## Step 2 — Start the Backend (FastAPI)

Open **Terminal 2**:

```bash
cd d:\Work_Dir\Projects\SAP_IBP\backend

# Install dependencies (first time only)
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected startup output:**
```
INFO     Loading LangGraph agent...
INFO     LangGraph agent loaded.
INFO     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Verify backend is healthy:**
```bash
curl http://localhost:8000/health
# {"status":"ok","service":"Billy API","version":"1.0.0"}
```

> The server pre-loads the LangGraph graph at startup. First request will be faster.

---

## Step 3 — Start the Frontend (React)

Open **Terminal 3**:

```bash
cd d:\Work_Dir\Projects\SAP_IBP\frontend
npm install     # first time only
npm run dev
```

**Expected output:**
```
  VITE v8.x  ready in 300ms
  ➜  Local:   http://localhost:5173/
```

Open **http://localhost:5173** in your browser.

The **green status dot** in the header confirms the frontend can reach the backend.

---

## Demo Script — Rehearsed Safe Questions

Use these questions in order. Each has a known, guaranteed answer from `inventory.csv`.

### Question 1 — Hero Demo (Upward Trend)
> **"Has inventory gone up or down over the last 6 months for Product A?"**

Expected Billy response:
> *"Inventory for Product A across all regions has risen 32.4% over the past 12 months, growing from 18,300 units in January 2025 to 24,200 units in December 2025 — a clear upward trend."*

---

### Question 2 — Region Filter
> **"Show me the inventory trend for Product A in North America."**

Expected Billy response:
> *"Inventory for Product A in North America has increased 32.4% from 10,200 units in January 2025 to 13,500 units in December 2025."*

---

### Question 3 — Downward Trend
> **"Is Product B running low in Europe?"**

Expected Billy response:
> *"Inventory for Product B in Europe has decreased 39.2% over 12 months, dropping from 12,000 units in January 2025 to 7,300 units in December 2025 — a significant depletion trend."*

---

### Question 4 — Stable / Flat Product
> **"Is Product C inventory stable?"**

Expected Billy response:
> *"Inventory for Product C has remained essentially flat over the past 12 months, with minor fluctuations between approximately 7,700 and 8,100 units in North America — no significant trend in either direction."*

---

### Question 5 — Graceful Failure (Out-of-Scope Refusal)
> **"What is the demand forecast for Product A next quarter?"**

Expected Billy response (instant, no LLM call):
> *"I'm Billy, your SAP IBP Inventory Assistant, and I'm currently optimised for inventory and stock-level queries only (Phase 1 MVP)..."*

This demonstrates the guardrail live — **intentionally use this question** to show Phase 2 scope awareness.

---

### Question 6 — Not-Found Graceful Failure
> **"What is the inventory for Product XYZ?"**

Expected Billy response:
> *"I couldn't find inventory data for 'Product XYZ' in our system. Could you double-check the product name?"*

Demonstrates the anti-hallucination guardrail — Billy refuses to invent a number.

---

## Fallback / Troubleshooting

| Symptom | Fix |
|---|---|
| Status dot is **red** | FastAPI is not running — check Terminal 2 |
| Response: "unable to reach language engine" | Ollama is down — check Terminal 1: `ollama serve` |
| Extremely slow response (>45s) | Model is still loading into VRAM — wait and retry |
| Model not found error | Run `ollama pull gemma4:12b` in Terminal 1 |
| CORS error in browser console | Ensure frontend runs on port 5173 or 3000 (both whitelisted) |
| `inventory.csv not found` error | Ensure `backend/inventory.csv` exists |
| 422 Unprocessable Entity | Message field is empty — type a question |

**Emergency fallback model** (if gemma4:12b fails to load):
```bash
# Kill Terminal 2, then:
set BILLY_MODEL=gemma4:e4b
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Architecture at a Glance

```
Browser (localhost:5173)
     │  POST /chat {"message": "..."}
     ▼
FastAPI (localhost:8000)              ← backend/main.py
     │  1. Pydantic v2 validates input
     │  2. Pre-flight scope check (0.1 ms)
     │  3. run_in_executor -> graph.invoke()
     ▼
LangGraph StateGraph                  ← backend/agent.py
     │  call_llm node -> ChatOllama
     │  get_inventory tool -> inventory.csv
     ▼
Ollama (localhost:11434)
     │  gemma4:12b (RTX 4060, 4-bit Q)
     ▼
{"response": "Inventory has risen 32.4%..."}
     │
     ▼
React chat bubble renders answer
```

---

## Environment Variables Reference

| Variable | Default | Effect |
|---|---|---|
| `BILLY_MODEL` | `gemma4:12b` | Switch LLM model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Point to remote Ollama |
| `VITE_API_URL` | `http://localhost:8000` | Frontend → backend URL |

---

## Monorepo Structure

```
SAP_IBP/
├── backend/                    ← Run uvicorn here
│   ├── main.py                 FastAPI app (POST /chat, GET /health)
│   ├── agent.py                LangGraph StateGraph + get_inventory tool
│   ├── prompts.py              System prompt + pre-flight scope guard
│   ├── inventory.csv           Mock SAP IBP data (96 rows, 4 SKUs, 2 regions)
│   ├── requirements.txt        Python dependencies
│   └── test_integration.py     Full integration test suite
│
├── frontend/                   ← Run npm run dev here
│   ├── src/
│   │   ├── App.jsx             Root — state, fetch, error handling
│   │   ├── main.jsx            Entry point
│   │   ├── index.css           Design system (dark theme, glassmorphism)
│   │   └── components/
│   │       ├── Header.jsx      Branding + live status dot
│   │       ├── ChatContainer.jsx  Scrollable history + welcome screen
│   │       ├── MessageBubble.jsx  3-variant chat bubbles
│   │       ├── TypingIndicator.jsx  Animated loader
│   │       └── ChatInput.jsx   Auto-resize textarea
│   ├── .env                    VITE_API_URL=http://localhost:8000
│   ├── vercel.json             SPA routing config
│   └── package.json
│
├── 1_data_engineering/         Source task artefacts (kept for reference)
├── 2_langgraph_agent/
├── 3_llm_guardrails/
├── 4_fastapi_backend/
├── 5_react_frontend/
│
└── RUNBOOK.md                  ← this file
```

---

*Billy MVP — EOD Demo Ready | 2026-07-07*
