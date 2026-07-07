# Task 4 — FastAPI REST Boundary Log

**Billy MVP — REST API Layer**  
File: `main.py` | Framework: FastAPI 0.115 | Pydantic: v2 | Python: 3.10+

---

## 1. Running the Server

### Start command
```bash
# From the 4_fastapi_backend directory:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

| Flag | Effect |
|---|---|
| `--reload` | Auto-restarts on code changes (dev only — remove in prod) |
| `--host 0.0.0.0` | Binds to all network interfaces (needed for LAN access during demo) |
| `--port 8000` | Standard non-privileged port; React expects this |

**Alternative (python entrypoint):**
```bash
python main.py
```

### Interactive API Docs
Once running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 2. Endpoint Reference

### `POST /chat`

**Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Has inventory gone up for Product A in North America?\"}"
```

**Windows PowerShell equivalent:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/chat" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"message": "Has inventory gone up for Product A in North America?"}'
```

---

### `GET /health`

```bash
curl http://localhost:8000/health
```

---

## 3. JSON Payload Schemas

### Successful inventory answer

```json
// Request
POST /chat
{
  "message": "Has inventory gone up for Product A in North America?"
}

// Response 200 OK
{
  "response": "Inventory for Product A in North America has risen 32.4% over the past 12 months, growing from 10,200 units in January 2025 to 13,500 units in December 2025 — a clear upward trend."
}
```

### Out-of-scope query (pre-flight refusal — no LLM call)

```json
// Request
POST /chat
{
  "message": "What is the demand forecast accuracy for Product B?"
}

// Response 200 OK  (intentionally 200, not 4xx — UI displays it as a message)
{
  "response": "I'm Billy, your SAP IBP Inventory Assistant, and I'm currently optimised for inventory and stock-level queries only (Phase 1 MVP). I can help with questions like 'What is the current stock for Product A?' or 'Has inventory gone up or down for Product B in Europe?'. For forecasts, scenario modelling, or other planning topics, please reach out to the wider planning team."
}
```

### Ollama engine down (graceful failure)

```json
// Response 200 OK  (polite fallback, never a raw 500)
{
  "response": "I am currently unable to reach my language engine. Please ensure the local Ollama server is running on port 11434."
}
```

### Validation error (bad request)

```json
// Request — missing required field
POST /chat
{}

// Response 422 Unprocessable Entity
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "message"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

---

## 4. Error Handling Architecture

Three layers ensure the demo never crashes:

```
POST /chat
   │
   ├─ [1] Pydantic v2 validation
   │       empty/missing message  → 422 (automatic)
   │       unknown fields         → 422 (extra="forbid")
   │
   ├─ [2] Pre-flight scope check  (is_out_of_scope)
   │       keyword match          → 200 OUT_OF_SCOPE_REPLY  (0.2 ms, zero LLM cost)
   │
   ├─ [3] try/except around graph.invoke()
   │       ConnectionError        → 200 _FALLBACK_OLLAMA_DOWN
   │       TimeoutError           → 200 _FALLBACK_OLLAMA_DOWN
   │       ValueError             → 200 _FALLBACK_DATA_ERROR
   │       Exception (catch-all)  → 200 _FALLBACK_UNEXPECTED
   │
   └─ [4] Global exception_handler (last resort)
           any escape             → 500 JSON {"response": ..., "error_code": ...}
```

> **Why return 200 for graceful errors?**  
> The React UI renders `response.response` as a chat message regardless of status.
> A 200 with a polite fallback message displays cleanly in the chat bubble.
> A 500 would require special error-state handling in the UI — unnecessary complexity for MVP.

---

## 5. CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # Create-React-App dev server
        "http://localhost:5173",    # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://*.vercel.app",     # Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)
```

The React frontend (Task 5) runs on **port 5173** (Vite) or **3000** (CRA). Both are whitelisted. The Vercel wildcard covers all preview deployment URLs without needing to know them in advance.

> **Security note:** For production, replace `"https://*.vercel.app"` with the exact Vercel domain.

---

## 6. Async Design: Why `run_in_executor`

`graph.invoke()` is a **synchronous blocking call** (LangGraph's default). Calling it directly inside an `async def` FastAPI handler would block the entire uvicorn event loop, making the API unresponsive to all other requests during the ~5–15 s Ollama inference time.

```python
# agent.py graph.invoke() is sync — must be offloaded
loop = asyncio.get_event_loop()
answer = await loop.run_in_executor(None, _blocking_call)
```

This keeps FastAPI's event loop free while the GPU is busy.

---

## 7. Architecture Connection

```
React UI (Task 5)
    │  POST /chat  {"message": "..."}
    ▼
FastAPI (main.py)
    │  1. Pydantic v2 validates ChatRequest
    │  2. is_out_of_scope() pre-flight check
    │  3. run_in_executor → graph.invoke()
    │      │
    │      ├─ call_llm node (ChatOllama / Gemma 4)
    │      └─ execute_tool node (get_inventory → inventory.csv)
    │  4. Extract last AIMessage.content
    │  5. Return ChatResponse {"response": "..."}
    ▼
React UI — renders response as chat bubble
```

---

## 8. Test Results (Verified Without Ollama)

| Test | Scenario | Expected | Result |
|---|---|---|---|
| 1 | GET /health | 200 `{status: ok}` | PASS |
| 2 | Empty message | 422 validation | PASS |
| 3 | Extra unknown field | 422 validation | PASS |
| 4 | Forecast question (out-of-scope) | 200 refusal | PASS (0.2 ms) |
| 5 | Missing `message` key | 422 validation | PASS |
| 6 | Response shape | `{response: str}` only | PASS |

---

*Generated: 2026-07-07 | Billy MVP — Task 4 Complete*
