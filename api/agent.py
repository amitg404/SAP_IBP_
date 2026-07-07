"""
Billy MVP — LangGraph Inventory Agent (Monorepo version)
All files co-located in backend/:
  agent.py             <- this file
  prompts.py           <- system prompt + pre-flight guard
  main.py              <- FastAPI wrapper
  inventory.csv        <- SAP IBP inventory data (existing)
  sales_history.csv    <- Sales orders 2023-2025 (1800 rows)
  purchase_orders.csv  <- Supplier purchase orders (480 rows)
  suppliers.csv        <- Supplier master data (60 rows)

Graph topology:
  START -> call_llm -> [tool_calls?] -> execute_tool -> call_llm -> END
                               <- [no tool calls]  -> END

Session memory:
  SESSION_MEMORY: dict[session_id -> deque[BaseMessage]] (last 5 turns)
  get_session_history(session_id) -> list[BaseMessage]
  append_to_session(session_id, human_msg, ai_msg)
"""

import json
import os
import sys
import pandas as pd
from collections import deque
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
CSV_PATH             = _HERE / "inventory.csv"
SALES_CSV_PATH       = _HERE / "sales_history.csv"
PURCHASE_CSV_PATH    = _HERE / "purchase_orders.csv"
SUPPLIER_CSV_PATH    = _HERE / "suppliers.csv"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from prompts import SYSTEM_PROMPT  # noqa: E402

# ── Model config — SINGLE source of truth ─────────────────────────────────────
# Set BILLY_MODEL env var to swap models without touching code.
# Tested:
#   qwen2.5:7b                    — current default (confirmed working)
#   gemma4:e2b                    — lighter, may OOM on <16GB RAM
#   gemini-3-flash-preview:cloud  — Ollama cloud model, pending test_model.py
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME      = os.getenv("BILLY_MODEL",     "qwen2.5:7b")
OLLAMA_API_KEY  = os.getenv("OLLAMA_API_KEY",  "")

# ── Session Memory Store ──────────────────────────────────────────────────────
# In-memory only — session-lifetime, no DB.
# dict[session_id -> deque[BaseMessage]], max 10 items = 5 full turns.
_MEMORY_WINDOW = 5

SESSION_MEMORY: dict[str, deque] = {}


def get_session_history(session_id: str) -> list:
    """Return stored message history for a session (empty list if new)."""
    return list(SESSION_MEMORY.get(session_id, []))


def append_to_session(session_id: str, human_msg: HumanMessage, ai_msg: AIMessage) -> None:
    """Append a completed turn to session store, auto-trimmed to window size."""
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = deque(maxlen=_MEMORY_WINDOW * 2)
    SESSION_MEMORY[session_id].append(human_msg)
    SESSION_MEMORY[session_id].append(ai_msg)


# ── State Schema ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── Shared CSV helpers ────────────────────────────────────────────────────────
def _load_csv() -> tuple:
    """Returns (df, None) on success or (None, error_json_str) on failure."""
    if not CSV_PATH.exists():
        return None, json.dumps({"error": f"Data file not found at '{CSV_PATH}'."})
    try:
        df = pd.read_csv(CSV_PATH, dtype={"inventory_qty": int})
        required = {"product_id", "product_name", "region", "period", "inventory_qty"}
        missing = required - set(df.columns)
        if missing:
            return None, json.dumps({"error": f"CSV missing columns: {missing}"})
        return df, None
    except Exception as exc:
        return None, json.dumps({"error": f"CSV load failed: {exc}"})


def _load_sales() -> tuple:
    """Load sales_history.csv. Returns (df, None) or (None, error_json_str)."""
    if not SALES_CSV_PATH.exists():
        return None, json.dumps({"error": "sales_history.csv not found. Run scratch/generate_mock_data.py first."})
    try:
        df = pd.read_csv(SALES_CSV_PATH)
        return df, None
    except Exception as exc:
        return None, json.dumps({"error": f"Sales CSV load failed: {exc}"})


def _load_purchase_orders() -> tuple:
    """Load purchase_orders.csv. Returns (df, None) or (None, error_json_str)."""
    if not PURCHASE_CSV_PATH.exists():
        return None, json.dumps({"error": "purchase_orders.csv not found. Run scratch/generate_mock_data.py first."})
    try:
        df = pd.read_csv(PURCHASE_CSV_PATH)
        return df, None
    except Exception as exc:
        return None, json.dumps({"error": f"Purchase orders CSV load failed: {exc}"})


def _load_suppliers() -> tuple:
    """Load suppliers.csv. Returns (df, None) or (None, error_json_str)."""
    if not SUPPLIER_CSV_PATH.exists():
        return None, json.dumps({"error": "suppliers.csv not found. Run scratch/generate_mock_data.py first."})
    try:
        df = pd.read_csv(SUPPLIER_CSV_PATH)
        return df, None
    except Exception as exc:
        return None, json.dumps({"error": f"Suppliers CSV load failed: {exc}"})


def _filter_product(df: pd.DataFrame, product_id: str) -> tuple:
    """Case-insensitive match on product_id OR product_name."""
    term = product_id.strip().lower()
    mask = (
        df["product_id"].str.strip().str.lower() == term
    ) | (
        df["product_name"].str.strip().str.lower() == term
    )
    filtered = df[mask].copy()
    if filtered.empty:
        available = sorted(df["product_name"].unique().tolist())
        return None, json.dumps({
            "error": f"No inventory data for '{product_id}'. Available: {available}"
        })
    return filtered, None


def _filter_region(df: pd.DataFrame, region: str, product_id: str) -> tuple:
    """Optional region filter. Returns (filtered_df, None) or (None, error_json)."""
    mask = df["region"].str.strip().str.lower() == region.strip().lower()
    result = df[mask].copy()
    if result.empty:
        avail = sorted(df["region"].unique().tolist())
        return None, json.dumps({
            "error": f"No data for '{product_id}' in region '{region}'. Available: {avail}"
        })
    return result, None


def _build_monthly(df: pd.DataFrame, region: str | None) -> pd.DataFrame:
    """Aggregate or select monthly series depending on region filter."""
    df = df.sort_values("period")
    if not region:
        return df.groupby("period")["inventory_qty"].sum().reset_index()
    return df[["period", "inventory_qty"]].reset_index(drop=True)


def _trend_stats(monthly: pd.DataFrame) -> dict:
    """Compute first/last/pct_change/trend_direction from a monthly series."""
    first_qty    = int(monthly.iloc[0]["inventory_qty"])
    last_qty     = int(monthly.iloc[-1]["inventory_qty"])
    pct_change   = round(((last_qty - first_qty) / first_qty) * 100, 1)
    return {
        "first_period":    str(monthly.iloc[0]["period"]),
        "first_qty":       first_qty,
        "last_period":     str(monthly.iloc[-1]["period"]),
        "last_qty":        last_qty,
        "pct_change":      pct_change,
        "trend_direction": (
            "increased" if pct_change > 2 else
            "decreased" if pct_change < -2 else
            "remained flat"
        ),
        "monthly_data": monthly.to_dict(orient="records"),
    }


# ── Tool: get_inventory ───────────────────────────────────────────────────────
@tool
def get_inventory(product_id: str, region: str = None, period: str = None) -> str:
    """
    Retrieve inventory data for a product from the SAP IBP mock dataset.

    ALWAYS call this tool for inventory questions. Never answer from memory.

    Args:
        product_id: Product identifier or name (case-insensitive).
                    Accepts product_id ('PROD-001') OR product_name ('Product A').
        region:     Optional region filter ('North America', 'Europe').
                    If omitted, aggregates across ALL regions.
        period:     Optional YYYY-MM period filter (e.g. '2025-06').
                    If omitted, returns all periods.

    Returns:
        JSON string with keys: product_name, product_id, region, period_range,
        first_period, first_qty, last_period, last_qty, pct_change,
        trend_direction, num_periods, monthly_data.
    """
    try:
        df, err = _load_csv()
        if err:
            return err

        filtered, err = _filter_product(df, product_id)
        if err:
            return err

        if region:
            filtered, err = _filter_region(filtered, region, product_id)
            if err:
                return err

        if period:
            filtered = filtered[filtered["period"] == period].copy()
            if filtered.empty:
                return json.dumps({"error": f"No data for period '{period}'."})

        monthly = _build_monthly(filtered, region)
        stats   = _trend_stats(monthly)

        return json.dumps({
            "product_name":  filtered["product_name"].iloc[0],
            "product_id":    filtered["product_id"].iloc[0],
            "region":        region if region else "All Regions (aggregated)",
            "period_range":  f"{stats['first_period']} to {stats['last_period']}",
            "num_periods":   len(monthly),
            **stats,
        })

    except Exception as exc:
        return json.dumps({"error": f"Tool execution failed: {exc}"})


# ── Tool: get_trend ───────────────────────────────────────────────────────────
@tool
def get_trend(product_id: str, region: str = None) -> str:
    """
    Calculate the inventory trend (% change over all available periods) for a product.

    Use this when the user asks about trends, going up/down, direction of change,
    or inventory movement over time.

    Args:
        product_id: Product identifier or name (case-insensitive).
        region:     Optional region filter. If omitted, aggregates all regions.

    Returns:
        JSON with: product_name, region, first_period, first_qty, last_period,
        last_qty, pct_change, trend_direction, monthly_data.
    """
    try:
        df, err = _load_csv()
        if err:
            return err

        filtered, err = _filter_product(df, product_id)
        if err:
            return err

        if region:
            filtered, err = _filter_region(filtered, region, product_id)
            if err:
                return err

        monthly = _build_monthly(filtered, region)
        stats   = _trend_stats(monthly)

        return json.dumps({
            "product_name": filtered["product_name"].iloc[0],
            "product_id":   filtered["product_id"].iloc[0],
            "region":       region if region else "All Regions (aggregated)",
            **stats,
        })

    except Exception as exc:
        return json.dumps({"error": f"Tool execution failed: {exc}"})


# ── Tool: compare_regions ─────────────────────────────────────────────────────
@tool
def compare_regions(product_id: str) -> str:
    """
    Compare inventory levels across all regions for a given product.

    Use this when the user asks to compare regions, see regional breakdown,
    or asks which region has more/less/higher/lower stock.

    Args:
        product_id: Product identifier or name (case-insensitive).

    Returns:
        JSON with: product_name, regions (list of {region, first_qty, last_qty,
        pct_change, trend_direction, monthly_data}).
    """
    try:
        df, err = _load_csv()
        if err:
            return err

        filtered, err = _filter_product(df, product_id)
        if err:
            return err

        regions_list = []
        for region_name, group in filtered.groupby("region"):
            monthly = group[["period", "inventory_qty"]].sort_values("period").reset_index(drop=True)
            stats   = _trend_stats(monthly)
            regions_list.append({"region": region_name, **stats})

        return json.dumps({
            "product_name": filtered["product_name"].iloc[0],
            "product_id":   filtered["product_id"].iloc[0],
            "regions":      regions_list,
        })

    except Exception as exc:
        return json.dumps({"error": f"Tool execution failed: {exc}"})


# ── Tool: aggregate ───────────────────────────────────────────────────────────
@tool
def aggregate(group_by: str) -> str:
    """
    Aggregate all inventory data grouped by region, product, or period.

    Use this when the user asks for totals, averages, summaries, or breakdowns
    across all products or regions (not scoped to a single product).

    Args:
        group_by: One of 'region', 'product', or 'period'.
                  - 'region'  -> total and average inventory per region
                  - 'product' -> total and average inventory per product
                  - 'period'  -> total inventory per month across all products/regions

    Returns:
        JSON with: group_by, rows (list of {group, total_qty, avg_qty}).
    """
    try:
        df, err = _load_csv()
        if err:
            return err

        group_by = group_by.strip().lower()
        valid = {"region", "product", "period"}
        if group_by not in valid:
            return json.dumps({"error": f"group_by must be one of {valid}. Got: '{group_by}'"})

        col = {"region": "region", "product": "product_name", "period": "period"}[group_by]

        grouped = (
            df.groupby(col)["inventory_qty"]
            .agg(total_qty="sum", avg_qty="mean")
            .reset_index()
        )
        grouped["avg_qty"] = grouped["avg_qty"].round(0).astype(int)

        rows = [
            {"group": row[col], "total_qty": int(row["total_qty"]), "avg_qty": int(row["avg_qty"])}
            for _, row in grouped.iterrows()
        ]

        return json.dumps({"group_by": group_by, "rows": rows})

    except Exception as exc:
        return json.dumps({"error": f"Tool execution failed: {exc}"})


# ── Tool: get_sales_trend ─────────────────────────────────────────────────────
@tool
def get_sales_trend(product_id: str = None, region: str = None, group_by: str = "period") -> str:
    """
    Analyse sales revenue and quantity trends from the sales_history dataset.

    Use this tool for ANY question about sales, revenue, orders sold, or sales
    performance — NOT for stock/inventory questions.

    Args:
        product_id: Optional product name or ID filter (e.g. 'Product A', 'PROD-001').
                    If omitted, aggregates ALL products.
        region:     Optional region filter (e.g. 'Europe', 'Asia Pacific').
                    If omitted, aggregates all regions.
        group_by:   How to group the result — 'period' (default, monthly timeline),
                    'product', 'region', or 'channel'.

    Returns:
        JSON with aggregated qty_sold, revenue, and trend data.
    """
    try:
        df, err = _load_sales()
        if err:
            return err

        # Optional product filter
        if product_id:
            term = product_id.strip().lower()
            mask = (
                df["product_id"].str.strip().str.lower() == term
            ) | (
                df["product_name"].str.strip().str.lower() == term
            )
            df = df[mask]
            if df.empty:
                available = sorted(df["product_name"].unique().tolist()) if not df.empty else []
                return json.dumps({"error": f"No sales data for '{product_id}'."})

        # Optional region filter
        if region:
            df = df[df["region"].str.strip().str.lower() == region.strip().lower()]
            if df.empty:
                return json.dumps({"error": f"No sales data for region '{region}'."})

        group_by = group_by.strip().lower()
        col_map = {
            "period": "period",
            "product": "product_name",
            "region": "region",
            "channel": "channel",
        }
        if group_by not in col_map:
            return json.dumps({"error": f"group_by must be one of {list(col_map)}. Got: '{group_by}'"})

        col = col_map[group_by]
        grouped = (
            df.groupby(col)
            .agg(qty_sold=("qty_sold", "sum"), revenue=("revenue", "sum"), returns=("returns_qty", "sum"))
            .reset_index()
            .sort_values(col)
        )

        rows = [
            {
                "group":     str(r[col]),
                "qty_sold":  int(r["qty_sold"]),
                "revenue":   round(float(r["revenue"]), 2),
                "returns":   int(r["returns"]),
            }
            for _, r in grouped.iterrows()
        ]

        # Compute overall trend stats if time-series
        trend = {}
        if group_by == "period" and len(rows) >= 2:
            first_rev = rows[0]["revenue"]
            last_rev  = rows[-1]["revenue"]
            pct = round(((last_rev - first_rev) / first_rev) * 100, 1) if first_rev else 0
            trend = {
                "first_period": rows[0]["group"],
                "last_period":  rows[-1]["group"],
                "revenue_change_pct": pct,
                "trend_direction": "increased" if pct > 2 else "decreased" if pct < -2 else "flat",
            }

        return json.dumps({
            "dataset":   "sales_history",
            "group_by":  group_by,
            "product_filter": product_id,
            "region_filter":  region,
            "rows":      rows,
            "trend":     trend,
        })

    except Exception as exc:
        return json.dumps({"error": f"get_sales_trend failed: {exc}"})


# ── Tool: get_purchase_orders ─────────────────────────────────────────────────
@tool
def get_purchase_orders(
    product_id: str = None,
    supplier_id: str = None,
    status: str = None,
    period: str = None,
) -> str:
    """
    Query purchase order (PO) data from the procurement dataset.

    Use for questions about supplier orders, procurement costs, lead times,
    delivery status, or open/received orders.

    Args:
        product_id:  Optional product name or ID (e.g. 'Product A', 'PROD-001').
        supplier_id: Optional supplier ID (e.g. 'SUP-001') or supplier name.
        status:      Optional PO status filter: 'Open', 'In Transit', 'Received',
                     'Partial', or 'Cancelled'.
        period:      Optional YYYY-MM period filter (e.g. '2024-06').

    Returns:
        JSON with PO summary: count, total_cost, avg_lead_time, and list of matching orders.
    """
    try:
        df, err = _load_purchase_orders()
        if err:
            return err

        if product_id:
            term = product_id.strip().lower()
            df = df[
                df["product_id"].str.strip().str.lower().eq(term)
                | df["product_name"].str.strip().str.lower().eq(term)
            ]

        if supplier_id:
            term = supplier_id.strip().lower()
            df = df[
                df["supplier_id"].str.strip().str.lower().eq(term)
                | df["supplier_name"].str.strip().str.lower().eq(term)
            ]

        if status:
            df = df[df["status"].str.strip().str.lower() == status.strip().lower()]

        if period:
            df = df[df["period"] == period.strip()]

        if df.empty:
            return json.dumps({"error": "No purchase orders match the specified filters."})

        summary = {
            "dataset":          "purchase_orders",
            "count":            len(df),
            "total_order_qty":  int(df["order_qty"].sum()),
            "total_received_qty": int(df["received_qty"].sum()),
            "total_cost":       round(float(df["total_cost"].sum()), 2),
            "avg_lead_time_days": round(float(df["lead_time_days"].mean()), 1),
            "status_breakdown": df["status"].value_counts().to_dict(),
            "orders": df.head(20).to_dict(orient="records"),  # cap at 20 for LLM context
        }
        return json.dumps(summary)

    except Exception as exc:
        return json.dumps({"error": f"get_purchase_orders failed: {exc}"})


# ── Tool: get_supplier_info ───────────────────────────────────────────────────
@tool
def get_supplier_info(
    supplier_id: str = None,
    region: str = None,
    category: str = None,
    min_rating: str = None,
) -> str:
    """
    Look up supplier master data — reliability ratings, lead times, costs, defect rates.

    Use for questions about suppliers, vendor performance, sourcing, contract expiry,
    or 'which supplier is best for X'.

    Args:
        supplier_id: Optional supplier ID (e.g. 'SUP-001') or name substring.
        region:      Optional region filter ('North America', 'Europe', 'Asia Pacific', etc.).
        category:    Optional product category ('Electronics', 'Raw Materials',
                     'Packaging', 'Chemicals', 'Machinery', 'Consumables').
        min_rating:  Optional minimum reliability rating filter ('A+', 'A', 'B+', 'B', 'C').

    Returns:
        JSON with supplier performance metrics.
    """
    try:
        df, err = _load_suppliers()
        if err:
            return err

        if supplier_id:
            term = supplier_id.strip().lower()
            df = df[
                df["supplier_id"].str.strip().str.lower().eq(term)
                | df["supplier_name"].str.strip().str.lower().str.contains(term)
            ]

        if region:
            df = df[df["region"].str.strip().str.lower() == region.strip().lower()]

        if category:
            df = df[df["category"].str.strip().str.lower() == category.strip().lower()]

        if min_rating:
            rating_order = {"A+": 5, "A": 4, "B+": 3, "B": 2, "C": 1}
            min_score = rating_order.get(min_rating.strip().upper(), 0)
            df = df[df["reliability_rating"].map(lambda r: rating_order.get(r, 0) >= min_score)]

        if df.empty:
            return json.dumps({"error": "No suppliers match the specified filters."})

        summary = {
            "dataset":    "suppliers",
            "count":      len(df),
            "avg_lead_time_days":     round(float(df["avg_lead_time_days"].mean()), 1),
            "avg_on_time_delivery_pct": round(float(df["on_time_delivery_pct"].mean()), 1),
            "avg_defect_rate_pct":     round(float(df["defect_rate_pct"].mean()), 2),
            "suppliers":  df.to_dict(orient="records"),
        }
        return json.dumps(summary)

    except Exception as exc:
        return json.dumps({"error": f"get_supplier_info failed: {exc}"})


# ── Tool registry ─────────────────────────────────────────────────────────────
TOOLS = [
    get_inventory,
    get_trend,
    compare_regions,
    aggregate,
    get_sales_trend,
    get_purchase_orders,
    get_supplier_info,
]
tool_node = ToolNode(TOOLS)


# ── LLM factory ───────────────────────────────────────────────────────────────
def _build_llm() -> ChatOllama:
    """
    Instantiate ChatOllama with .bind_tools().
    MODEL_NAME is the single config point — set via BILLY_MODEL env var.
    """
    try:
        llm = ChatOllama(
            model=MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            num_predict=512,
        )
        return llm.bind_tools(TOOLS)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at '{OLLAMA_BASE_URL}'. "
            f"Start: ollama serve && ollama pull {MODEL_NAME}. Error: {exc}"
        ) from exc


# ── Graph Nodes ───────────────────────────────────────────────────────────────
def call_llm(state: AgentState) -> AgentState:
    """
    Node: invoke LLM with full message history.
    Injects SystemMessage guardrail on first call (idempotent).
    """
    messages = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    try:
        llm = _build_llm()
        response = llm.invoke(messages)
        return {"messages": [response]}

    except RuntimeError as exc:
        return {"messages": [AIMessage(content=(
            f"Unable to reach AI engine. Ensure Ollama is running with '{MODEL_NAME}'. Detail: {exc}"
        ))]}
    except Exception as exc:  # noqa: BLE001
        return {"messages": [AIMessage(content=(
            f"Unexpected error: {exc}. Please try again."
        ))]}


def should_use_tool(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "execute_tool"
    return END


# ── Graph Assembly ────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
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
    print(f"  Model  : {MODEL_NAME}")
    print(f"  URL    : {OLLAMA_BASE_URL}")
    print(f"  CSV    : {CSV_PATH}")
    print(f"  Tools  : {[t.name for t in TOOLS]}")
    print("=" * 65)

    test_cases = [
        ("Upward trend",       "Has inventory gone up for Product A?"),
        ("Region filter",      "Show inventory trend for Product B in North America."),
        ("Region comparison",  "Compare Product A across regions"),
        ("Aggregate",          "Show total inventory by product"),
        ("Not-found guard",    "What is the inventory for Widget XYZ?"),
        ("Out-of-scope",       "What is the demand forecast for Product A?"),
    ]

    for i, (label, query) in enumerate(test_cases, 1):
        print(f"\n[{i}] {label}\n  Q: {query}\n  {'-'*58}")
        try:
            result = graph.invoke({"messages": [HumanMessage(content=query)]})
            print(f"  A: {result['messages'][-1].content}")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print("\n" + "=" * 65)
    print("  Smoke test complete.")
    print("=" * 65)
