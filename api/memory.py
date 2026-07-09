"""
Conversation Memory Backend — Supabase + local fallback.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  get_session_history(session_id)                        │
  │    -> Supabase: SELECT last N rows ORDER BY created_at  │
  │    -> Local fallback: in-memory deque                   │
  │                                                         │
  │  append_to_session(session_id, human_msg, ai_msg)       │
  │    -> Supabase: INSERT 2 rows (role=human, role=ai)     │
  │    -> Local fallback: append to deque                   │
  │                                                         │
  │  clear_session(session_id)                              │
  │    -> Supabase: DELETE WHERE session_id = ?             │
  │    -> Local fallback: del from dict                     │
  └─────────────────────────────────────────────────────────┘

Supabase table (run this SQL in your Supabase dashboard):

  CREATE TABLE conversation_memory (
    id         BIGSERIAL PRIMARY KEY,
    session_id TEXT        NOT NULL,
    role       TEXT        NOT NULL CHECK (role IN ('human', 'ai')),
    content    TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  );

  CREATE INDEX idx_conv_mem_session
    ON conversation_memory (session_id, created_at);

  -- Optional: auto-purge rows older than 30 days (requires pg_cron extension)
  -- SELECT cron.schedule('purge-old-memory', '0 3 * * *',
  --   $$DELETE FROM conversation_memory WHERE created_at < NOW() - INTERVAL '30 days'$$);

Env vars required (add to .env and Vercel dashboard):
  SUPABASE_URL  = https://<your-project>.supabase.co
  SUPABASE_KEY  = <anon or service_role key>
"""

import logging
import os
from collections import deque

from langchain_core.messages import AIMessage, HumanMessage

log = logging.getLogger("billy.memory")

# -- Config -------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
TABLE_NAME   = "conversation_memory"
WINDOW       = 5   # retain last N complete turns (1 turn = 1 human + 1 ai row)

# -- Backend selection --------------------------------------------------------
_USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_KEY)
_supabase     = None

if _USE_SUPABASE:
    try:
        from supabase import create_client  # noqa: PLC0415
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        log.info("Supabase memory backend active (%s)", SUPABASE_URL)
    except ImportError:
        log.warning("supabase package not installed; falling back to in-memory store.")
        _USE_SUPABASE = False
    except Exception as exc:
        log.warning("Supabase connection failed (%s); falling back to in-memory store.", exc)
        _USE_SUPABASE = False
else:
    log.info("SUPABASE_URL not set — using in-memory session store (local dev mode).")

# -- Local in-memory fallback -------------------------------------------------
_local_store: dict[str, deque] = {}


# =============================================================================
# Public API
# =============================================================================

def get_session_history(session_id: str) -> list:
    """
    Return the last WINDOW turns as a list of LangChain BaseMessage objects.
    Ordered oldest → newest.
    """
    if _USE_SUPABASE and _supabase:
        return _supabase_get(session_id)
    return _local_get(session_id)


def append_to_session(
    session_id: str,
    human_msg: HumanMessage,
    ai_msg: AIMessage,
) -> None:
    """Persist a completed turn (human + AI message pair) to the store."""
    if _USE_SUPABASE and _supabase:
        _supabase_append(session_id, human_msg, ai_msg)
    else:
        _local_append(session_id, human_msg, ai_msg)


def clear_session(session_id: str) -> None:
    """Delete all history for a given session (e.g. on user logout or reset)."""
    if _USE_SUPABASE and _supabase:
        _supabase_clear(session_id)
    else:
        _local_clear(session_id)


def backend_name() -> str:
    """Returns 'supabase' or 'in-memory' — useful for health check endpoint."""
    return "supabase" if (_USE_SUPABASE and _supabase) else "in-memory"


# =============================================================================
# Supabase backend
# =============================================================================

def _supabase_get(session_id: str) -> list:
    try:
        # Fetch last WINDOW*2 rows (each turn = 2 rows: human + ai)
        resp = (
            _supabase.table(TABLE_NAME)
            .select("role, content, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(WINDOW * 2)
            .execute()
        )
        rows = list(reversed(resp.data or []))  # flip back to asc order
        messages = []
        for row in rows:
            if row["role"] == "human":
                messages.append(HumanMessage(content=row["content"]))
            elif row["role"] == "ai":
                messages.append(AIMessage(content=row["content"]))
        return messages
    except Exception as exc:
        log.error("Supabase get_session_history failed: %s — returning empty history.", exc)
        return []


def _supabase_append(
    session_id: str,
    human_msg: HumanMessage,
    ai_msg: AIMessage,
) -> None:
    try:
        _supabase.table(TABLE_NAME).insert([
            {"session_id": session_id, "role": "human", "content": human_msg.content},
            {"session_id": session_id, "role": "ai",    "content": ai_msg.content},
        ]).execute()
        # Keep only the latest WINDOW*2 rows — purge older rows
        _supabase_trim(session_id)
    except Exception as exc:
        log.error("Supabase append_to_session failed: %s", exc)


def _supabase_trim(session_id: str) -> None:
    """Delete rows beyond the sliding window limit."""
    try:
        resp = (
            _supabase.table(TABLE_NAME)
            .select("id")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .execute()
        )
        all_ids = [r["id"] for r in (resp.data or [])]
        ids_to_delete = all_ids[WINDOW * 2:]  # keep newest WINDOW*2
        if ids_to_delete:
            _supabase.table(TABLE_NAME).delete().in_("id", ids_to_delete).execute()
    except Exception as exc:
        log.warning("Supabase trim failed (non-critical): %s", exc)


def _supabase_clear(session_id: str) -> None:
    try:
        _supabase.table(TABLE_NAME).delete().eq("session_id", session_id).execute()
    except Exception as exc:
        log.error("Supabase clear_session failed: %s", exc)


# =============================================================================
# In-memory fallback backend
# =============================================================================

def _local_get(session_id: str) -> list:
    return list(_local_store.get(session_id, []))


def _local_append(
    session_id: str,
    human_msg: HumanMessage,
    ai_msg: AIMessage,
) -> None:
    if session_id not in _local_store:
        _local_store[session_id] = deque(maxlen=WINDOW * 2)
    _local_store[session_id].append(human_msg)
    _local_store[session_id].append(ai_msg)


def _local_clear(session_id: str) -> None:
    _local_store.pop(session_id, None)
