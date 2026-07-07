"""
Billy MVP — FastAPI REST Boundary
Task 4: POST /chat endpoint wrapping the LangGraph agent

Architecture:
    React UI  →  POST /chat  →  LangGraph graph  →  Ollama (Gemma 4)
                                     ↕
                              get_inventory tool
                                     ↕
                              inventory.csv (Task 1)

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# ── sys.path wiring ───────────────────────────────────────────────────────────
# All source folders sit at the SAP_IBP root. Insert them so local imports work
# regardless of where uvicorn is launched from.
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "2_langgraph_agent"))
sys.path.insert(0, str(_ROOT / "3_llm_guardrails"))

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from prompts import OUT_OF_SCOPE_REPLY, is_out_of_scope

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("billy.api")

# ── Fallback messages (shown to the React UI on failure) ──────────────────────
_FALLBACK_OLLAMA_DOWN = (
    "I am currently unable to reach my language engine. "
    "Please ensure the local Ollama server is running on port 11434."
)
_FALLBACK_DATA_ERROR = (
    "I couldn't find inventory data for that request. "
    "Please try asking about a specific product — for example: "
    "'Show inventory trend for Product A in North America.'"
)
_FALLBACK_UNEXPECTED = (
    "Something went wrong on my end. Please try again in a moment."
)


# ── Lazy graph import (deferred so startup doesn't fail if Ollama is down) ────
_graph = None


def _get_graph():
    """Import and cache the compiled LangGraph on first call."""
    global _graph  # noqa: PLW0603
    if _graph is None:
        log.info("Loading LangGraph agent…")
        from agent import graph  # noqa: PLC0415  (intentional deferred import)
        _graph = graph
        log.info("LangGraph agent loaded ✓")
    return _graph


# ── Lifespan: warm up graph at startup ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the graph on server start so the first request isn't slow."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _get_graph)
    except Exception as exc:  # noqa: BLE001
        log.warning("Graph pre-load failed (Ollama may be down): %s", exc)
    yield
    log.info("Billy API shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Billy — SAP IBP Inventory Assistant API",
    description=(
        "REST boundary for the Billy MVP. "
        "Accepts natural-language inventory questions, "
        "routes them through a LangGraph agent backed by a local Gemma 4 model, "
        "and returns a human-readable conversational answer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow the React dev server (Vite default :5173, CRA default :3000)
# and any Vercel preview URL. Tighten to specific origins before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Create-React-App dev server
        "http://localhost:5173",   # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://*.vercel.app",    # Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ── Pydantic v2 Schemas ───────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    """Incoming payload from the React frontend."""

    model_config = {"extra": "forbid"}  # reject unknown fields

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language inventory question from the planner.",
        examples=["Has inventory gone up for Product A in the last 6 months?"],
    )


class ChatResponse(BaseModel):
    """Outgoing payload — clean string only, no internal LangGraph metadata."""

    response: str = Field(
        ...,
        description="Human-readable answer from Billy.",
    )


class ErrorResponse(BaseModel):
    """Structured error payload — always returned instead of raw 500."""

    response: str
    error_code: str


# ── Helper: run blocking graph.invoke in a thread pool ───────────────────────
async def _invoke_graph(user_message: str) -> str:
    """
    Runs the synchronous LangGraph graph.invoke() in FastAPI's default
    thread-pool executor so it does not block the async event loop.

    Returns the final AIMessage content string.
    """
    graph = _get_graph()

    def _blocking_call():
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_message)]}
        )
        return result["messages"][-1].content

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _blocking_call)


# ── POST /chat ────────────────────────────────────────────────────────────────
@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask Billy an inventory question",
    responses={
        200: {"model": ChatResponse, "description": "Successful answer"},
        422: {"description": "Validation error — bad request payload"},
        503: {"model": ErrorResponse, "description": "Ollama engine unavailable"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main endpoint. Flow:

    1. Pre-flight scope check  →  instant refusal if out-of-scope
    2. invoke LangGraph graph  →  LLM + tool call + synthesis
    3. Return ChatResponse     →  {"response": "<conversational string>"}

    All failure paths return a structured JSON response — never a raw 500.
    """
    user_message = request.message.strip()
    t0 = time.perf_counter()
    log.info("Received query: %r", user_message[:120])

    # ── Layer 1: Pre-flight scope check (zero LLM cost) ──────────────────────
    if is_out_of_scope(user_message):
        log.info("Pre-flight: out-of-scope query rejected in %.1f ms",
                 (time.perf_counter() - t0) * 1000)
        return ChatResponse(response=OUT_OF_SCOPE_REPLY)

    # ── Layer 2: LangGraph invocation ─────────────────────────────────────────
    try:
        answer = await _invoke_graph(user_message)
        elapsed = (time.perf_counter() - t0) * 1000
        log.info("Response generated in %.0f ms", elapsed)
        return ChatResponse(response=answer)

    except (ConnectionError, ConnectionRefusedError, OSError) as exc:
        # Ollama server unreachable
        log.error("Ollama connection error: %s", exc)
        return ChatResponse(response=_FALLBACK_OLLAMA_DOWN)

    except TimeoutError as exc:
        log.error("Ollama request timed out: %s", exc)
        return ChatResponse(response=_FALLBACK_OLLAMA_DOWN)

    except ValueError as exc:
        # Tool returned bad data / product not found leaked through
        log.warning("Data/value error: %s", exc)
        return ChatResponse(response=_FALLBACK_DATA_ERROR)

    except Exception as exc:  # noqa: BLE001
        # Catch-all — log full traceback, return polite message
        log.exception("Unexpected error processing query: %s", exc)
        return ChatResponse(response=_FALLBACK_UNEXPECTED)


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
async def health():
    """Lightweight liveness probe — used by React to detect if backend is up."""
    return {"status": "ok", "service": "Billy API", "version": "1.0.0"}


# ── Global exception handler (last-resort 500 guard) ─────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches any unhandled exception that escapes the route handler.
    Ensures the React frontend always receives valid JSON, never an empty 500.
    """
    log.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"response": _FALLBACK_UNEXPECTED, "error_code": "INTERNAL_ERROR"},
    )


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
