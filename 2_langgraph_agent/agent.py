"""
Billy MVP — LangGraph Inventory Agent
Task 2: AI Orchestration Layer

Graph topology:
  START → call_llm → [tool_calls?] → execute_tool → call_llm → END
                              └─ [no tool calls] → END

System prompt sourced from: ../3_llm_guardrails/prompts.py
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Annotated, TypedDict

import sys
from pathlib import Path as _Path
# Pull the centralized guardrails prompt (Task 3) onto sys.path
sys.path.insert(0, str(_Path(__file__).parent.parent / "3_llm_guardrails"))
from prompts import SYSTEM_PROMPT  # noqa: E402

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── Constants ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
CSV_PATH = _HERE.parent / "1_data_engineering" / "inventory.csv"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("BILLY_MODEL", "gemma4:12b")

# SYSTEM_PROMPT imported above from 3_llm_guardrails/prompts.py


# ── State Schema (TypedDict — strict LangGraph v1.2+ requirement) ─────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── Tool: get_inventory ───────────────────────────────────────────────────────
@tool
def get_inventory(product_id: str, region: str = None) -> str:
    """
    Retrieve inventory data for a product from the SAP IBP mock dataset.

    Use this tool for ANY question about inventory quantities, trends, or
    stock levels. Always call this tool — never answer from memory.

    Args:
        product_id: Product identifier or human-readable name to look up.
                    Accepts product_id (e.g. 'PROD-001') OR product_name
                    (e.g. 'Product A'). Matching is case-insensitive.
        region:     Optional geographic filter (e.g. 'North America', 'Europe').
                    If omitted, data is aggregated across ALL regions.

    Returns:
        JSON string with keys: product_name, product_id, region,
        period_range, first_period, first_qty, last_period, last_qty,
        pct_change, trend_direction, num_periods, monthly_data.
        On failure: JSON string with key 'error' describing the problem.
    """
    try:
        # ── Guard: CSV must exist ─────────────────────────────────────────────
        if not CSV_PATH.exists():
            return json.dumps({
                "error": (
                    f"Data file not found at '{CSV_PATH}'. "
                    "Ensure inventory.csv from Task 1 is present."
                )
            })

        df = pd.read_csv(CSV_PATH, dtype={"inventory_qty": int})

        # ── Case-insensitive product match (id OR name) ───────────────────────
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

        # ── Optional region filter ────────────────────────────────────────────
        if region:
            region_mask = (
                filtered["region"].str.strip().str.lower()
                == region.strip().lower()
            )
            region_df = filtered[region_mask].copy()
            if region_df.empty:
                available_regions = sorted(filtered["region"].unique().tolist())
                return json.dumps({
                    "error": (
                        f"No data found for '{product_id}' in region '{region}'. "
                        f"Available regions: {available_regions}"
                    )
                })
            filtered = region_df

        # ── Sort chronologically (YYYY-MM string sort is safe) ───────────────
        filtered = filtered.sort_values("period")

        # ── Aggregate across regions when no region filter ────────────────────
        if not region:
            monthly = (
                filtered.groupby("period")["inventory_qty"]
                .sum()
                .reset_index()
            )
        else:
            monthly = filtered[["period", "inventory_qty"]].reset_index(drop=True)

        # ── Trend calculation ─────────────────────────────────────────────────
        first_qty   = int(monthly.iloc[0]["inventory_qty"])
        last_qty    = int(monthly.iloc[-1]["inventory_qty"])
        first_period = str(monthly.iloc[0]["period"])
        last_period  = str(monthly.iloc[-1]["period"])

        pct_change = round(((last_qty - first_qty) / first_qty) * 100, 1)

        if pct_change > 2:
            trend_direction = "increased"
        elif pct_change < -2:
            trend_direction = "decreased"
        else:
            trend_direction = "remained flat"

        return json.dumps({
            "product_name":  filtered["product_name"].iloc[0],
            "product_id":    filtered["product_id"].iloc[0],
            "region":        region if region else "All Regions (aggregated)",
            "period_range":  f"{first_period} to {last_period}",
            "first_period":  first_period,
            "first_qty":     first_qty,
            "last_period":   last_period,
            "last_qty":      last_qty,
            "pct_change":    pct_change,
            "trend_direction": trend_direction,
            "num_periods":   len(monthly),
            "monthly_data":  monthly.to_dict(orient="records"),
        })

    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Tool execution failed: {exc}"})


# ── Tool registry (used by both ToolNode and LLM binding) ────────────────────
TOOLS = [get_inventory]
tool_node = ToolNode(TOOLS)


# ── LLM factory (called lazily to allow import without Ollama running) ────────
def _build_llm() -> ChatOllama:
    """
    Instantiate ChatOllama with tool bindings.
    Raises RuntimeError with a helpful message if Ollama is unreachable.
    """
    try:
        llm = ChatOllama(
            model=MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,   # low temp → deterministic, factual answers
            num_predict=512,   # cap tokens for snappy API responses
        )
        return llm.bind_tools(TOOLS)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Cannot connect to Ollama at '{OLLAMA_BASE_URL}'. "
            f"Run: ollama serve  and  ollama pull {MODEL_NAME}. "
            f"Original error: {exc}"
        ) from exc


# ── Graph Nodes ───────────────────────────────────────────────────────────────
def call_llm(state: AgentState) -> AgentState:
    """
    Node — invoke the LLM with the full message history.

    Prepends the system prompt on every call so the guardrails persist
    across multi-turn conversations. Handles Ollama downtime gracefully.
    """
    messages = state["messages"]

    # Ensure system prompt is always first
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    try:
        llm_with_tools = _build_llm()
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    except RuntimeError as exc:
        # Ollama is down — return polite fallback so the UI never hangs
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I'm sorry, I'm currently unable to reach my AI engine. "
                        "Please ensure the Ollama server is running and the model "
                        f"'{MODEL_NAME}' has been pulled. "
                        f"Technical detail: {exc}"
                    )
                )
            ]
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I encountered an unexpected error while processing your "
                        f"request. Please try again. (Detail: {exc})"
                    )
                )
            ]
        }


def should_use_tool(state: AgentState) -> str:
    """
    Conditional edge function — routes after call_llm.

    Returns 'execute_tool' if the LLM emitted a tool call,
    or END if a final text answer was produced.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "execute_tool"
    return END


# ── Graph Assembly ────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """
    Compile and return the Billy LangGraph state machine.

    Topology:
        START
          └─► call_llm
                ├─► [has tool_calls] ─► execute_tool ─► call_llm  (loop)
                └─► [no tool_calls]  ─► END
    """
    builder = StateGraph(AgentState)

    builder.add_node("call_llm", call_llm)
    builder.add_node("execute_tool", tool_node)

    builder.set_entry_point("call_llm")

    builder.add_conditional_edges(
        "call_llm",
        should_use_tool,
        {
            "execute_tool": "execute_tool",
            END: END,
        },
    )

    # After tool execution → back to LLM to produce the final answer
    builder.add_edge("execute_tool", "call_llm")

    return builder.compile()


# ── Public compiled graph (imported by FastAPI in Task 4) ─────────────────────
graph = build_graph()


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Billy MVP — LangGraph Agent Standalone Smoke Test")
    print("=" * 65)

    test_cases = [
        # (description, query)
        (
            "DEMO Q — upward trend",
            "Has inventory gone up or down over the last 6 months for Product A?",
        ),
        (
            "Downward trend with region filter",
            "Show me the inventory trend for Product B in North America.",
        ),
        (
            "Flat/stable product",
            "Is Product C inventory stable?",
        ),
        (
            "Not-found graceful failure",
            "What is the inventory for a product called 'Widget XYZ'?",
        ),
    ]

    for idx, (label, query) in enumerate(test_cases, 1):
        print(f"\n[{idx}] {label}")
        print(f"  Query   : {query}")
        print(f"  {'─' * 58}")
        try:
            result = graph.invoke({"messages": [HumanMessage(content=query)]})
            answer = result["messages"][-1].content
            print(f"  Response: {answer}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR   : {exc}")

    print("\n" + "=" * 65)
    print("  Smoke test complete.")
    print("=" * 65)
