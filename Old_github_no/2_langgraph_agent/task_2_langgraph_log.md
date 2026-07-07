# Task 2 — LangGraph Agent Log

**Billy MVP — AI Orchestration Layer**  
File: `agent.py` | Framework: LangGraph 1.2.8 | Model: Gemma 4 12B via Ollama

---

## 1. Graph Architecture

### Topology Diagram

```
START
  └──► [call_llm]
           │
           ├── tool_calls present? ──YES──► [execute_tool] ──► [call_llm]  ← (loop)
           │
           └── tool_calls absent?  ──NO───► END
```

### Nodes

| Node | Function | Responsibility |
|---|---|---|
| `call_llm` | `call_llm(state)` | Invokes ChatOllama with full message history + system prompt. Catches Ollama downtime and returns a polite AIMessage fallback instead of crashing. |
| `execute_tool` | `ToolNode(TOOLS)` | LangGraph prebuilt node. Executes whichever tool the LLM called, wraps the result in a `ToolMessage`, appends it to state. |

### Edges

| From | To | Condition |
|---|---|---|
| `START` | `call_llm` | Always (entry point) |
| `call_llm` | `execute_tool` | `last_message.tool_calls` is non-empty |
| `call_llm` | `END` | No tool calls — final answer produced |
| `execute_tool` | `call_llm` | Always — return to LLM for synthesis |

### Conditional Routing Logic (`should_use_tool`)

```python
def should_use_tool(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "execute_tool"
    return END
```

The loop terminates naturally when the LLM produces a plain text answer (no `tool_calls`). Maximum loop depth for MVP is 2 iterations (one tool call round-trip).

---

## 2. State Schema

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

`add_messages` is a LangGraph reducer that **appends** new messages to the existing list on every state update — it does not overwrite. This preserves the full conversation history across nodes.

Message types used:
- `SystemMessage` — Billy's guardrail prompt (injected by `call_llm` if absent)
- `HumanMessage` — incoming user query
- `AIMessage` — LLM response (may contain `tool_calls`)
- `ToolMessage` — tool execution result (produced by `ToolNode`)

---

## 3. `get_inventory` Tool — Input/Output Schema

### Inputs

| Parameter | Type | Required | Description |
|---|---|---|---|
| `product_id` | `str` | ✅ Yes | Product ID (e.g. `PROD-001`) OR product name (e.g. `Product A`). Case-insensitive. |
| `region` | `str` | ❌ Optional | Geographic region filter. If omitted, aggregates across all regions. |

### Output (success)

```json
{
  "product_name":    "Product A",
  "product_id":      "PROD-001",
  "region":          "North America",
  "period_range":    "2025-01 to 2025-12",
  "first_period":    "2025-01",
  "first_qty":       10200,
  "last_period":     "2025-12",
  "last_qty":        13500,
  "pct_change":      32.4,
  "trend_direction": "increased",
  "num_periods":     12,
  "monthly_data":    [{"period": "2025-01", "inventory_qty": 10200}, ...]
}
```

### Output (failure)

```json
{
  "error": "No inventory data found for 'Widget XYZ'. Available products: ['Product A', 'Product B', 'Product C', 'Product D']"
}
```

### Failure modes handled

| Scenario | Response |
|---|---|
| CSV file missing | `{"error": "Data file not found at ..."}` |
| Product not in CSV | `{"error": "No inventory data found for '...' Available products: [...]"}` |
| Region not found for product | `{"error": "No data found for '...' in region '...'. Available regions: [...]"}` |
| Unexpected exception | `{"error": "Tool execution failed: <exc message>"}` |

---

## 4. Model Configuration

| Setting | Value | Reason |
|---|---|---|
| Model | `gemma4:12b` | Encoder-free arch; native agentic tool-calling |
| `base_url` | `http://localhost:11434` | Standard Ollama port (override via `OLLAMA_BASE_URL` env var) |
| `temperature` | `0.1` | Deterministic, factual inventory answers |
| `num_predict` | `512` | Caps token output for fast API responses |
| Tool binding | `.bind_tools(TOOLS)` | Forces LLM to emit structured JSON tool calls |

**Override model via env var:**
```bash
set BILLY_MODEL=gemma4:e4b   # lighter model for low VRAM
set OLLAMA_BASE_URL=http://192.168.1.10:11434  # remote Ollama
```

---

## 5. Instructions for Task 4 (FastAPI)

### Import & Invoke

```python
# In fastapi_app.py
from agent import graph  # imports the pre-compiled StateGraph
from langchain_core.messages import HumanMessage

async def run_agent(user_message: str) -> str:
    result = graph.invoke({
        "messages": [HumanMessage(content=user_message)]
    })
    # Final answer is always the last message
    return result["messages"][-1].content
```

### For async FastAPI endpoints (recommended)

```python
import asyncio

async def run_agent_async(user_message: str) -> str:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,  # uses default thread pool
        lambda: graph.invoke({"messages": [HumanMessage(content=user_message)]})
    )
    return result["messages"][-1].content
```

> **Why `run_in_executor`?**  
> `graph.invoke()` is synchronous (blocking). Wrapping it in `run_in_executor` prevents it from blocking FastAPI's async event loop.

### Payload contract

- **Input:** `{"messages": [HumanMessage(content="<user question>")]}`
- **Output key:** `result["messages"][-1].content` → plain string → return in `{"response": "..."}` Pydantic model

### Error handling in FastAPI layer

Wrap the `graph.invoke()` call in a try/except. The agent already returns a polite `AIMessage` when Ollama is down, so the FastAPI layer should never need to expose raw tracebacks.

```python
try:
    answer = await run_agent_async(user_message)
except Exception:
    answer = "Billy is temporarily unavailable. Please try again shortly."
```

---

## 6. Architecture Connection

```
inventory.csv (Task 1)
       │ read at tool-call time via pandas
       ▼
get_inventory tool
       │ bound to LLM via .bind_tools()
       ▼
LangGraph StateGraph  ◄────── FastAPI POST /chat (Task 4)
  call_llm ↔ execute_tool          │
       │                           ▼
       └─── final AIMessage   React UI (Task 5)
```

---

*Generated: 2026-07-07 | Billy MVP — Task 2 Complete*
