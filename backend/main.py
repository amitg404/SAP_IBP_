"""
Billy MVP — Backend (Monorepo)
Unified backend: FastAPI + LangGraph + Guardrails + Mock Data

All source files live in this single `backend/` directory:
  main.py       — FastAPI app  (POST /chat, GET /health)
  agent.py      — LangGraph StateGraph + get_inventory tool
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
from contextlib import asynccontextmanager
from pathlib import Path

# ── All sibling modules are in the same directory ────────────────────────────
_BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
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
    "and the model has been pulled (ollama pull gemma4:12b)."
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
        "conversational, data-grounded answers via LangGraph + Gemma 4."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # CRA dev
        "http://localhost:5173",    # Vite dev
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://*.vercel.app",     # Vercel preview/prod
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ── Pydantic v2 Schemas ───────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    """Incoming payload from the React frontend."""
    model_config = {"extra": "forbid"}

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language inventory question from the planner.",
        examples=["Has inventory gone up for Product A in the last 6 months?"],
    )


class ChatResponse(BaseModel):
    """Outgoing payload — clean string only, no LangGraph internals."""
    response: str = Field(..., description="Human-readable answer from Billy.")


class ErrorResponse(BaseModel):
    """Structured error payload."""
    response: str
    error_code: str


# ── Helper: run blocking graph.invoke in thread pool ─────────────────────────
async def _invoke_graph(user_message: str) -> str:
    """
    Offloads synchronous graph.invoke() to thread pool so it does not
    block FastAPI's async event loop during the ~5-30s Ollama inference.
    """
    graph = _get_graph()

    def _blocking():
        result = graph.invoke({"messages": [HumanMessage(content=user_message)]})
        return result["messages"][-1].content

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _blocking)


# ── POST /chat ────────────────────────────────────────────────────────────────
@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask Billy an inventory question",
    responses={
        200: {"model": ChatResponse},
        422: {"description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Engine unavailable"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Flow:
    1. Pre-flight scope check  -> instant refusal (0ms, no LLM cost)
    2. Invoke LangGraph graph  -> LLM + tool call + synthesis
    3. Return ChatResponse     -> {"response": "<conversational string>"}

    All failure paths return structured JSON — never a raw 500.
    """
    user_message = request.message.strip()
    t0 = time.perf_counter()
    log.info("Received query: %r", user_message[:120])

    # Layer 1: pre-flight scope check
    if is_out_of_scope(user_message):
        log.info("Pre-flight: out-of-scope rejected in %.1f ms",
                 (time.perf_counter() - t0) * 1000)
        return ChatResponse(response=OUT_OF_SCOPE_REPLY)

    # Layer 2: LangGraph invocation with full error handling
    try:
        answer = await _invoke_graph(user_message)
        log.info("Response generated in %.0f ms", (time.perf_counter() - t0) * 1000)
        return ChatResponse(response=answer)

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


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health", summary="Liveness probe")
async def health():
    """Lightweight health check — polled every 15s by the React frontend."""
    return {"status": "ok", "service": "Billy API", "version": "1.0.0"}


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
