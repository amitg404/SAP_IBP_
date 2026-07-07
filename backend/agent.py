"""
Billy MVP — LangGraph Inventory Agent (Monorepo version)
All files now co-located in backend/:
  agent.py      <- this file
  prompts.py    <- system prompt + pre-flight guard
  inventory.csv <- mock SAP IBP data
  main.py       <- FastAPI wrapper

Graph topology:
  START -> call_llm -> [tool_calls?] -> execute_tool -> call_llm -> END
                               <- [no tool calls]  -> END
"""

import json
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── Paths (all sibling files in same backend/ directory) ─────────────────────
_HERE = Path(__file__).parent
CSV_PATH = _HERE / "inventory.csv"

# prompts.py is a sibling — ensure importable
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from prompts import SYSTEM_PROMPT  # noqa: E402

# ── Model config (override via environment variables) ─────────────────────────
# Supported models (local Ollama):
#   gemma4:12b   — recommended (RTX 4060 8GB, 4-bit quantised)
#   gemma4:e4b   — lighter alternative for <8GB VRAM
#   qwen3:8b     — alternative encoder-free model with strong tool-calling
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME      = os.getenv("BILLY_MODEL",     "gemma4:12b")


# ── State Schema ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    """
    Strict TypedDict — LangGraph v1.2+ requirement.
    add_messages reducer appends (never overwrites) to preserve history.
    """
    messages: Annotated[list, add_messages]


# ── Tool: get_inventory ───────────────────────────────────────────────────────
@tool
def get_inventory(product_id: str, region: str = None) -> str:
    """
    Retrieve inventory data for a product from the SAP IBP mock dataset.

    ALWAYS call this tool for inventory questions. Never answer from memory.

    Args:
        product_id: Product identifier or name (case-insensitive).
                    Accepts product_id ('PROD-001') OR product_name ('Product A').
        region:     Optional region filter ('North America', 'Europe').
                    If omitted, aggregates across ALL regions.

    Returns:
        JSON string. Success keys: product_name, product_id, region,
        period_range, first_period, first_qty, last_period, last_qty,
        pct_change, trend_direction, num_periods, monthly_data.
        Error key: 'error' with a descriptive message.

    CSV schema (inventory.csv):
        product_id   : str  — e.g. 'PROD-001'
        product_name : str  — e.g. 'Product A'
        region       : str  — e.g. 'North America', 'Europe'
        period       : str  — YYYY-MM format e.g. '2025-01'
        inventory_qty: int  — stock count
    """
    try:
        # Guard: CSV must exist
        if not CSV_PATH.exists():
            return json.dumps({
                "error": (
                    f"Data file not found at '{CSV_PATH}'. "
                    "Ensure inventory.csv is present in the backend/ directory."
                )
            })

        df = pd.read_csv(CSV_PATH, dtype={"inventory_qty": int})

        # Validate required columns are present
        required_cols = {"product_id", "product_name", "region", "period", "inventory_qty"}
        missing = required_cols - set(df.columns)
        if missing:
            return json.dumps({"error": f"CSV is missing required columns: {missing}"})

        # Case-insensitive match on product_id OR product_name
        term = product_id.strip().lower()
        mask = (
            df["product_id"].str.strip().str.lower() == term
        ) | (
            df["product_name"].str.strip().str.lower() == term
        )
        filtered = df[mask].copy()

        if filtered.empty:
            available = sorted(df["product_name"].unique().tolist())
            return json.dumps({
                "error": (
                    f"No inventory data found for '{product_id}'. "
                    f"Available products: {available}"
                )
            })

        # Optional region filter
        if region:
            region_mask = (
                filtered["region"].str.strip().str.lower()
                == region.strip().lower()
            )
            region_df = filtered[region_mask].copy()
            if region_df.empty:
                avail_regions = sorted(filtered["region"].unique().tolist())
                return json.dumps({
                    "error": (
                        f"No data for '{product_id}' in region '{region}'. "
                        f"Available regions: {avail_regions}"
                    )
                })
            filtered = region_df

        # Sort chronologically — YYYY-MM lexicographic sort is safe
        filtered = filtered.sort_values("period")

        # Aggregate across regions when no filter applied
        if not region:
            monthly = (
                filtered.groupby("period")["inventory_qty"]
                .sum()
                .reset_index()
            )
        else:
            monthly = filtered[["period", "inventory_qty"]].reset_index(drop=True)

        # Trend calculation
        first_qty    = int(monthly.iloc[0]["inventory_qty"])
        last_qty     = int(monthly.iloc[-1]["inventory_qty"])
        first_period = str(monthly.iloc[0]["period"])
        last_period  = str(monthly.iloc[-1]["period"])
        pct_change   = round(((last_qty - first_qty) / first_qty) * 100, 1)

        if pct_change > 2:
            trend_direction = "increased"
        elif pct_change < -2:
            trend_direction = "decreased"
        else:
            trend_direction = "remained flat"

        return json.dumps({
            "product_name":    filtered["product_name"].iloc[0],
            "product_id":      filtered["product_id"].iloc[0],
            "region":          region if region else "All Regions (aggregated)",
            "period_range":    f"{first_period} to {last_period}",
            "first_period":    first_period,
            "first_qty":       first_qty,
            "last_period":     last_period,
            "last_qty":        last_qty,
            "pct_change":      pct_change,
            "trend_direction": trend_direction,
            "num_periods":     len(monthly),
            "monthly_data":    monthly.to_dict(orient="records"),
        })

    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Tool execution failed: {exc}"})


# ── Tool registry ─────────────────────────────────────────────────────────────
TOOLS = [get_inventory]
tool_node = ToolNode(TOOLS)


# ── LLM factory ───────────────────────────────────────────────────────────────
def _build_llm() -> ChatOllama:
    """
    Instantiate ChatOllama with .bind_tools() for structured tool-call JSON.

    Supported models via BILLY_MODEL env var:
      gemma4:12b  (default) — Gemma 4 12B Q4, native tool-calling
      gemma4:e4b            — Gemma 4 E4B, lighter
      qwen3:8b              — Qwen3 8B, strong tool-calling alternative

    temperature=0.1 enforces deterministic, factual inventory answers.
    num_predict=512 caps output for fast API responses.
    """
    try:
        llm = ChatOllama(
            model=MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            num_predict=512,
        )
        return llm.bind_tools(TOOLS)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Cannot connect to Ollama at '{OLLAMA_BASE_URL}'. "
            f"Start with: ollama serve && ollama pull {MODEL_NAME}. "
            f"Error: {exc}"
        ) from exc


# ── Graph Nodes ───────────────────────────────────────────────────────────────
def call_llm(state: AgentState) -> AgentState:
    """
    Node: invoke LLM with full message history.
    Injects SystemMessage guardrail on first call (idempotent).
    Catches Ollama downtime — returns polite AIMessage fallback.
    """
    messages = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    try:
        llm = _build_llm()
        response = llm.invoke(messages)
        return {"messages": [response]}

    except RuntimeError as exc:
        return {
            "messages": [AIMessage(content=(
                "I'm sorry, I'm currently unable to reach my AI engine. "
                f"Please ensure Ollama is running with model '{MODEL_NAME}'. "
                f"Detail: {exc}"
            ))]
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "messages": [AIMessage(content=(
                f"Unexpected error while processing your request: {exc}. "
                "Please try again."
            ))]
        }


def should_use_tool(state: AgentState) -> str:
    """
    Conditional edge: route to execute_tool if LLM emitted tool_calls,
    otherwise terminate at END with the final text answer.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "execute_tool"
    return END


# ── Graph Assembly ────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """
    Compile the Billy LangGraph state machine.

    Topology:
        START -> call_llm
                   |- [tool_calls] -> execute_tool -> call_llm  (loop until done)
                   `- [no calls]   -> END
    """
    builder = StateGraph(AgentState)
    builder.add_node("call_llm", call_llm)
    builder.add_node("execute_tool", tool_node)
    builder.set_entry_point("call_llm")
    builder.add_conditional_edges(
        "call_llm",
        should_use_tool,
        {"execute_tool": "execute_tool", END: END},
    )
    builder.add_edge("execute_tool", "call_llm")
    return builder.compile()


# Public compiled graph — imported by main.py
graph = build_graph()


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Billy MVP - LangGraph Agent Smoke Test")
    print("=" * 65)
    print(f"  Model     : {MODEL_NAME}")
    print(f"  Ollama URL: {OLLAMA_BASE_URL}")
    print(f"  CSV path  : {CSV_PATH}")
    print("=" * 65)

    test_cases = [
        ("DEMO Q - upward trend",
         "Has inventory gone up or down over the last 6 months for Product A?"),
        ("Region filter",
         "Show me the inventory trend for Product B in North America."),
        ("Stable product",
         "Is Product C inventory stable?"),
        ("Not-found guard",
         "What is the inventory for Widget XYZ?"),
        ("Out-of-scope (should be refused)",
         "What is the demand forecast for Product A next quarter?"),
    ]

    for i, (label, query) in enumerate(test_cases, 1):
        print(f"\n[{i}] {label}")
        print(f"  Q: {query}")
        print(f"  {'-' * 58}")
        try:
            result = graph.invoke({"messages": [HumanMessage(content=query)]})
            print(f"  A: {result['messages'][-1].content}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")

    print("\n" + "=" * 65)
    print("  Smoke test complete.")
    print("=" * 65)
