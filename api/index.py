"""
Billy MVP — Backend (Monorepo)
Unified backend: FastAPI + LangGraph + Guardrails + Mock Data

All source files live in this single `backend/` directory:
  main.py       — FastAPI app  (POST /chat, GET /health)
  agent.py      — LangGraph StateGraph + inventory tools
  prompts.py    — System prompt + pre-flight scope guard
  inventory.csv — Mock SAP IBP data (Task 1)

Run:
  cd backend
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import sys
import time
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

# ── All sibling modules are in the same directory ────────────────────────────
_BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(_BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(_BACKEND_DIR / ".env")

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from prompts import OUT_OF_SCOPE_REPLY, is_out_of_scope

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("billy.api")

# ── Fallback messages ─────────────────────────────────────────────────────────
_FALLBACK_OLLAMA_DOWN = (
    "I am currently unable to reach my language engine. "
    "Please ensure the local Ollama server is running on port 11434 "
    "and the model has been pulled."
)
_FALLBACK_DATA_ERROR = (
    "I couldn't find inventory data for that request. "
    "Please try asking about a specific product — for example: "
    "'Show inventory trend for Product A in North America.'"
)
_FALLBACK_UNEXPECTED = (
    "Something went wrong on my end. Please try again in a moment."
)

# ── Lazy graph import ─────────────────────────────────────────────────────────
_graph = None


def _get_graph():
    """Import and cache the compiled LangGraph on first call."""
    global _graph  # noqa: PLW0603
    if _graph is None:
        log.info("Loading LangGraph agent...")
        from agent import graph  # noqa: PLC0415
        _graph = graph
        log.info("LangGraph agent loaded.")
    return _graph


# ── Lifespan: warm up graph at startup ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the graph so the first real request isn't slow."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _get_graph)
    except Exception as exc:  # noqa: BLE001
        log.warning("Graph pre-load skipped (Ollama may be down): %s", exc)
    yield
    log.info("Billy API shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Billy - SAP IBP Inventory Assistant API",
    description=(
        "REST boundary for the Billy MVP. "
        "Accepts natural-language inventory questions and returns "
        "conversational, data-grounded answers via LangGraph + Ollama."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ── Pydantic v2 Schemas ───────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    """Incoming payload from the React frontend."""
    model_config = {"extra": "ignore"}  # allow extra fields for forward compat

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language supply chain question.",
        examples=["Has inventory gone up for Product A in the last 6 months?"],
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Session identifier for conversation memory.",
    )
    # FIX 1: client owns persona state -- no server-side SESSION_PERSONA dict needed.
    # Frontend sends back the active_persona it received in the last ChatResponse.
    active_persona: str | None = Field(
        default=None,
        description="Active persona from prior response ('blake', 'chris', or None).",
    )


class ChartData(BaseModel):
    """Optional chart payload — present only when the query implies a visualisation."""
    chart_type: Literal["line", "bar", "pie"]
    title: str
    x_key: str
    y_key: str
    data: list[dict]


class ChatResponse(BaseModel):
    """Outgoing payload -- text answer + optional chart + persona metadata."""
    response: str = Field(..., description="Human-readable answer.")
    chart: ChartData | None = Field(default=None)
    # FIX 1: stateless persona -- client stores this and echoes it next request.
    persona: str = Field(default="router", description="Which persona answered.")
    active_persona: str | None = Field(
        default=None,
        description="Persona to route to on next request (None = start fresh).",
    )


class ErrorResponse(BaseModel):
    """Structured error payload."""
    response: str
    error_code: str


# ── Chart extraction ──────────────────────────────────────────────────────────
def _extract_chart(messages: list) -> ChartData | None:
    """
    Inspect tool messages in the graph result to derive chart data.
    Returns a ChartData if a visualisable tool was called, else None.

    Tool → chart type mapping:
      get_trend        → line  (time-series trend)
      compare_regions  → bar   (region comparison)
      aggregate        → bar/pie depending on group_by
      get_inventory    → line  (when monthly_data present)
    """
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue

        tool_name = getattr(msg, "name", "") or ""
        try:
            payload = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue

        if "error" in payload:
            continue

        # get_trend → line chart (period vs qty)
        if tool_name == "get_trend" and "monthly_data" in payload:
            return ChartData(
                chart_type="line",
                title=f"Inventory Trend — {payload.get('product_name', '')} ({payload.get('region', '')})",
                x_key="period",
                y_key="inventory_qty",
                data=payload["monthly_data"],
            )

        # compare_regions → bar chart (region vs last_qty)
        if tool_name == "compare_regions" and "regions" in payload:
            bar_data = [
                {"region": r["region"], "inventory_qty": r["last_qty"]}
                for r in payload["regions"]
            ]
            return ChartData(
                chart_type="bar",
                title=f"Regional Comparison — {payload.get('product_name', '')} (Latest Period)",
                x_key="region",
                y_key="inventory_qty",
                data=bar_data,
            )

        # aggregate → bar (product/period) or pie (region)
        if tool_name == "aggregate" and "rows" in payload:
            group_by = payload.get("group_by", "")
            chart_type = "pie" if group_by == "region" else "bar"
            return ChartData(
                chart_type=chart_type,
                title=f"Inventory by {group_by.title()}",
                x_key="group",
                y_key="total_qty",
                data=payload["rows"],
            )

        # get_inventory with monthly_data → line chart
        if tool_name == "get_inventory" and "monthly_data" in payload:
            region_label = payload.get("region", "")
            return ChartData(
                chart_type="line",
                title=f"Inventory — {payload.get('product_name', '')} ({region_label})",
                x_key="period",
                y_key="inventory_qty",
                data=payload["monthly_data"],
            )

        # get_sales_trend → line (period), bar (product/channel), pie (region)
        if tool_name == "get_sales_trend" and "rows" in payload:
            rows     = payload["rows"]
            group_by = payload.get("group_by", "period")
            prod     = payload.get("product_filter") or "All Products"
            region   = payload.get("region_filter") or "All Regions"

            if group_by == "period":
                return ChartData(
                    chart_type="line",
                    title=f"Sales Revenue — {prod} ({region})",
                    x_key="group",
                    y_key="revenue",
                    data=rows,
                )
            if group_by == "region":
                return ChartData(
                    chart_type="pie",
                    title=f"Revenue by Region — {prod}",
                    x_key="group",
                    y_key="revenue",
                    data=rows,
                )
            # product or channel → bar
            return ChartData(
                chart_type="bar",
                title=f"Revenue by {group_by.title()}",
                x_key="group",
                y_key="revenue",
                data=rows,
            )

        # get_purchase_orders → bar chart by status breakdown
        if tool_name == "get_purchase_orders" and "status_breakdown" in payload:
            status_rows = [
                {"group": k, "total_qty": v}
                for k, v in payload["status_breakdown"].items()
            ]
            prod_label = payload.get("product_filter") or "All Products"
            return ChartData(
                chart_type="bar",
                title=f"Purchase Orders by Status — {prod_label}",
                x_key="group",
                y_key="total_qty",
                data=status_rows,
            )

        # get_supplier_info → bar chart by supplier on-time delivery
        if tool_name == "get_supplier_info" and "suppliers" in payload:
            sup_rows = payload["suppliers"]
            if sup_rows:
                chart_rows = [
                    {
                        "group": s.get("supplier_name", s.get("supplier_id", "")),
                        "on_time_pct": s.get("on_time_delivery_pct", 0),
                    }
                    for s in sup_rows[:12]  # cap to avoid chart overflow
                ]
                return ChartData(
                    chart_type="bar",
                    title="Supplier On-Time Delivery %",
                    x_key="group",
                    y_key="on_time_pct",
                    data=chart_rows,
                )

    return None


# ── Helper: run blocking graph.invoke in thread pool ─────────────────────────
async def _invoke_graph(
    user_message: str,
    session_id: str,
    active_persona: str | None,
) -> tuple[str, ChartData | None, str, str | None]:
    """
    Offloads synchronous graph.invoke() to thread pool.
    Seeds graph with session history for conversation memory.
    """
    from memory import get_session_history, append_to_session  # noqa: PLC0415
    graph = _get_graph()

    def _blocking():
        history = get_session_history(session_id)
        human   = HumanMessage(content=user_message)
        seed    = history + [human]

        log.info(
            "Session %s — history=%d msgs, persona=%s",
            session_id[:8], len(history), active_persona,
        )

        result   = graph.invoke({
            "messages":         seed,
            "active_persona":   active_persona,
            "session_id":       session_id,
            "escalation_count": 0,
        })
        messages = result["messages"]

        answer_msg = messages[-1]
        answer     = answer_msg.content if hasattr(answer_msg, "content") else str(answer_msg)
        chart      = _extract_chart(messages)

        next_persona = result.get("active_persona")
        who_answered = next_persona or "router"

        append_to_session(session_id, human, AIMessage(content=answer))
        return answer, chart, who_answered, next_persona

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _blocking)


# ── POST /api/chat ────────────────────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse, summary="Main chat endpoint")
async def api_chat(request: ChatRequest) -> ChatResponse:
    """
    Flow:
    1. Pre-flight scope check  -> instant refusal (0ms, no LLM cost)
    2. Load session history    -> seed graph with last N messages
    3. Invoke LangGraph graph  -> LLM + tool call + synthesis
    4. Extract chart payload   -> from tool messages (if visualisable query)
    5. Persist turn to memory  -> for next request in same session
    6. Return ChatResponse     -> {response: str, chart: ChartData | None}
    """
    user_message = request.message.strip()
    session_id   = request.session_id
    t0 = time.perf_counter()
    log.info("Received query: %r (session=%s)", user_message[:120], session_id[:8])

    # Layer 1: pre-flight scope check
    if is_out_of_scope(user_message):
        log.info("Pre-flight: out-of-scope rejected in %.1f ms",
                 (time.perf_counter() - t0) * 1000)
        return ChatResponse(response=OUT_OF_SCOPE_REPLY)

    # Layer 2: LangGraph invocation with full error handling
    try:
        answer, chart, who_answered, next_persona = await _invoke_graph(
            user_message, session_id, request.active_persona
        )
        log.info(
            "Response in %.0f ms (chart=%s, persona=%s)",
            (time.perf_counter() - t0) * 1000, chart is not None, who_answered,
        )
        return ChatResponse(
            response=answer,
            chart=chart,
            persona=who_answered,
            active_persona=next_persona,
        )

    except (ConnectionError, ConnectionRefusedError, OSError) as exc:
        log.error("Ollama connection error: %s", exc)
        return ChatResponse(response=_FALLBACK_OLLAMA_DOWN)

    except TimeoutError as exc:
        log.error("Request timed out: %s", exc)
        return ChatResponse(response=_FALLBACK_OLLAMA_DOWN)

    except ValueError as exc:
        log.warning("Data/value error: %s", exc)
        return ChatResponse(response=_FALLBACK_DATA_ERROR)

    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error: %s", exc)
        return ChatResponse(response=_FALLBACK_UNEXPECTED)


# ── GET /api/health ───────────────────────────────────────────────────────────
@app.get("/api/health", summary="Liveness probe")
async def health():
    from memory import backend_name  # noqa: PLC0415
    return {"status": "ok", "service": "Billy API", "version": "2.0.0", "memory": backend_name()}


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Last-resort catch — ensures React always receives valid JSON."""
    log.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"response": _FALLBACK_UNEXPECTED, "error_code": "INTERNAL_ERROR"},
    )


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
