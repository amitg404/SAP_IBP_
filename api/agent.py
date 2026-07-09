"""
Multi-Agent Supply Chain AI -- LangGraph Backend

Agents:
  Router  -- Concierge, classifies intent and routes (or answers directly).
  Blake   -- Data & Analytics (inventory, sales, POs, suppliers, charts).
  Chris   -- Forecasting & Planning (demand forecast, seasonality, what-if).

Graph topology (Hybrid Concierge + Escalation, stateless-persona):

  START
    |
    v
  route_entry  <-- checks active_persona in state (passed from frontend)
    |
    +-- (None)  --> call_router --> after_router
    |                                   |
    |              HANDLE_DIRECTLY  CLARIFY_USER  ROUTE_TO_BLAKE  ROUTE_TO_CHRIS
    |                   |               |               |               |
    |                  END             END           call_blake      call_chris
    |
    +-- (blake) --> call_blake --> should_continue_blake
    |                                   |
    |                  END / execute_blake_tools (loop) / escalate
    |                                                         |
    |                                              handle_escalation
    |                                              (count check -> clarify or re-route)
    |
    +-- (chris) --> call_chris --> should_continue_chris
                                        |
                         END / execute_chris_tools (loop) / escalate

FIX 1: active_persona lives in AgentState (passed from client each request).
        No in-process SESSION_PERSONA dict needed -- fully stateless-compatible.
FIX 2: Robust token parsing via substring match with priority order.
FIX 3: escalation_count in AgentState; capped at 2; forces clarify on overflow.
FIX 4: Catch-all escalation rule in Blake/Chris system prompts (in prompts.py).
FIX 6: System prompt injected fresh per node call -- no voice bleed.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# -- Paths --------------------------------------------------------------------
_HERE             = Path(__file__).parent
CSV_PATH          = _HERE / "inventory.csv"
SALES_CSV_PATH    = _HERE / "sales_history.csv"
PURCHASE_CSV_PATH = _HERE / "purchase_orders.csv"
SUPPLIER_CSV_PATH = _HERE / "suppliers.csv"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from prompts import (  # noqa: E402
    ROUTER_SYSTEM_PROMPT,
    BLAKE_SYSTEM_PROMPT,
    CHRIS_SYSTEM_PROMPT,
)

# -- Model config -------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME      = os.getenv("BILLY_MODEL",     "gemma4:31b-cloud")
CHRIS_MODEL     = os.getenv("CHRIS_MODEL",     "deepseek-v3.2:cloud")
ROUTER_MODEL    = os.getenv("ROUTER_MODEL",    "nemotron-3-nano:30b-cloud")
OLLAMA_API_KEY  = os.getenv("OLLAMA_API_KEY",  "")

# -- Session Memory -- delegated to memory.py (Supabase or local fallback) ---
from memory import get_session_history, append_to_session, clear_session  # noqa: E402


# -- State Schema -------------------------------------------------------------
# FIX 1: active_persona and escalation_count are graph-state fields passed in
#         from the client, not server-side dict lookups.
class AgentState(TypedDict):
    messages:         Annotated[list, add_messages]
    active_persona:   str | None   # "blake", "chris", or None (router)
    session_id:       str
    escalation_count: int          # FIX 3: ping-pong guard


# =============================================================================
# CSV HELPERS
# =============================================================================

def _load_csv() -> tuple:
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
    if not SALES_CSV_PATH.exists():
        return None, json.dumps({"error": "sales_history.csv not found."})
    try:
        return pd.read_csv(SALES_CSV_PATH), None
    except Exception as exc:
        return None, json.dumps({"error": f"Sales CSV load failed: {exc}"})


def _load_purchase_orders() -> tuple:
    if not PURCHASE_CSV_PATH.exists():
        return None, json.dumps({"error": "purchase_orders.csv not found."})
    try:
        return pd.read_csv(PURCHASE_CSV_PATH), None
    except Exception as exc:
        return None, json.dumps({"error": f"Purchase orders CSV load failed: {exc}"})


def _load_suppliers() -> tuple:
    if not SUPPLIER_CSV_PATH.exists():
        return None, json.dumps({"error": "suppliers.csv not found."})
    try:
        return pd.read_csv(SUPPLIER_CSV_PATH), None
    except Exception as exc:
        return None, json.dumps({"error": f"Suppliers CSV load failed: {exc}"})


def _filter_product(df: pd.DataFrame, product_id: str) -> tuple:
    term = product_id.strip().lower()
    mask = (
        df["product_id"].str.strip().str.lower() == term
    ) | (
        df["product_name"].str.strip().str.lower() == term
    ) | (
        df["product_name"].str.strip().str.lower().str.contains(term, na=False)
    )
    filtered = df[mask].copy()
    if filtered.empty:
        avail = sorted(df["product_name"].unique().tolist())
        return None, json.dumps({
            "error": f"No data for '{product_id}'. Available: {avail}"
        })
    return filtered, None


def _filter_region(df: pd.DataFrame, region: str, product_id: str) -> tuple:
    mask   = df["region"].str.strip().str.lower() == region.strip().lower()
    result = df[mask].copy()
    if result.empty:
        avail = sorted(df["region"].unique().tolist())
        return None, json.dumps({
            "error": f"No data for '{product_id}' in '{region}'. Available: {avail}"
        })
    return result, None


def _build_monthly(df: pd.DataFrame, region: str | None) -> pd.DataFrame:
    df = df.sort_values("period")
    if not region:
        return df.groupby("period")["inventory_qty"].sum().reset_index()
    return df[["period", "inventory_qty"]].reset_index(drop=True)


def _trend_stats(monthly: pd.DataFrame) -> dict:
    first_qty  = int(monthly.iloc[0]["inventory_qty"])
    last_qty   = int(monthly.iloc[-1]["inventory_qty"])
    pct_change = round(((last_qty - first_qty) / first_qty) * 100, 1) if first_qty else 0
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


# =============================================================================
# BLAKE TOOLS
# =============================================================================

@tool
def get_inventory(product_id: str, region: str = None, period: str = None) -> str:
    """
    Retrieve inventory data for a product from the SAP IBP mock dataset.
    ALWAYS call this tool for inventory questions. Never answer from memory.
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
            "product_name": filtered["product_name"].iloc[0],
            "product_id":   filtered["product_id"].iloc[0],
            "region":       region if region else "All Regions (aggregated)",
            "period_range": f"{stats['first_period']} to {stats['last_period']}",
            "num_periods":  len(monthly),
            **stats,
        })
    except Exception as exc:
        return json.dumps({"error": f"Tool execution failed: {exc}"})


@tool
def get_trend(product_id: str, region: str = None) -> str:
    """
    Calculate inventory trend (% change) for a product over all available periods.
    """
    return get_inventory.invoke({"product_id": product_id, "region": region})


@tool
def compare_regions(product_id: str) -> str:
    """
    Compare inventory levels for a product across all regions for the latest period.
    """
    try:
        df, err = _load_csv()
        if err:
            return err
        filtered, err = _filter_product(df, product_id)
        if err:
            return err
        last_period  = filtered["period"].max()
        latest       = filtered[filtered["period"] == last_period]
        regions_data = sorted(
            [{"region": row["region"], "last_qty": int(row["inventory_qty"]), "period": last_period}
             for _, row in latest.iterrows()],
            key=lambda x: x["last_qty"], reverse=True
        )
        return json.dumps({
            "product_name": filtered["product_name"].iloc[0],
            "product_id":   filtered["product_id"].iloc[0],
            "period":       last_period,
            "regions":      regions_data,
        })
    except Exception as exc:
        return json.dumps({"error": f"compare_regions failed: {exc}"})


@tool
def aggregate(group_by: str = "product") -> str:
    """
    Aggregate total inventory across the dataset.
    group_by must be one of: 'region', 'product', 'period'.
    """
    try:
        df, err = _load_csv()
        if err:
            return err
        mapping = {"region": "region", "product": "product_name", "period": "period"}
        col = mapping.get(group_by.lower())
        if not col:
            return json.dumps({"error": f"group_by must be 'region', 'product', or 'period'."})
        grouped = df.groupby(col)["inventory_qty"].sum().reset_index()
        grouped.columns = ["group", "total_qty"]
        grouped = grouped.sort_values("total_qty", ascending=False)
        return json.dumps({
            "group_by": group_by,
            "rows":     grouped.to_dict(orient="records"),
            "total":    int(grouped["total_qty"].sum()),
        })
    except Exception as exc:
        return json.dumps({"error": f"aggregate failed: {exc}"})


@tool
def get_sales_trend(
    product_id: str = None,
    region: str = None,
    group_by: str = "period",
) -> str:
    """
    Retrieve sales revenue and quantity trends from sales_history.csv.
    group_by: 'period' (time series), 'product', 'region', or 'channel'.
    """
    try:
        df, err = _load_sales()
        if err:
            return err
        if product_id:
            term = product_id.strip().lower()
            mask = (df["product_id"].str.strip().str.lower() == term) | \
                   (df["product_name"].str.strip().str.lower().str.contains(term, na=False))
            df = df[mask].copy()
            if df.empty:
                return json.dumps({"error": f"No sales data for product '{product_id}'."})
        if region:
            df = df[df["region"].str.strip().str.lower() == region.strip().lower()].copy()
            if df.empty:
                return json.dumps({"error": f"No sales data for region '{region}'."})
        col_map   = {"period": "period", "product": "product_name", "region": "region", "channel": "channel"}
        group_col = col_map.get(group_by.lower(), "period")
        grouped   = df.groupby(group_col).agg(revenue=("revenue", "sum"), qty_sold=("qty_sold", "sum")).reset_index()
        grouped.columns = ["group", "revenue", "qty_sold"]
        grouped["revenue"]  = grouped["revenue"].round(2)
        grouped["qty_sold"] = grouped["qty_sold"].astype(int)
        grouped = grouped.sort_values("group" if group_by == "period" else "revenue",
                                      ascending=(group_by == "period"))
        return json.dumps({
            "group_by":       group_by,
            "product_filter": product_id,
            "region_filter":  region,
            "rows":           grouped.to_dict(orient="records"),
            "total_revenue":  round(float(grouped["revenue"].sum()), 2),
            "total_qty":      int(grouped["qty_sold"].sum()),
        })
    except Exception as exc:
        return json.dumps({"error": f"get_sales_trend failed: {exc}"})


@tool
def get_purchase_orders(
    product_id: str = None,
    supplier_id: str = None,
    status: str = None,
    period: str = None,
) -> str:
    """Query purchase order data: status, cost, lead times, quantities."""
    try:
        df, err = _load_purchase_orders()
        if err:
            return err
        if product_id:
            term = product_id.strip().lower()
            df = df[(df["product_id"].str.strip().str.lower() == term) |
                    (df["product_name"].str.strip().str.lower().str.contains(term, na=False))].copy()
        if supplier_id:
            df = df[df["supplier_id"].str.strip().str.lower() == supplier_id.strip().lower()].copy()
        if status:
            df = df[df["status"].str.strip().str.lower() == status.strip().lower()].copy()
        if period:
            df = df[df["period"] == period].copy()
        if df.empty:
            return json.dumps({"error": "No purchase orders match the given filters."})
        return json.dumps({
            "product_filter":   product_id,
            "total_orders":     len(df),
            "total_order_qty":  int(df["order_qty"].sum()),
            "total_cost":       round(float(df["total_cost"].sum()), 2),
            "avg_lead_time":    round(float(df["lead_time_days"].mean()), 1),
            "status_breakdown": df.groupby("status")["order_qty"].sum().to_dict(),
            "cost_by_supplier": df.groupby("supplier_name")["total_cost"].sum().round(2).to_dict(),
            "sample_orders":    df.head(5).to_dict(orient="records"),
        })
    except Exception as exc:
        return json.dumps({"error": f"get_purchase_orders failed: {exc}"})


@tool
def get_supplier_info(
    supplier_id: str = None,
    region: str = None,
    category: str = None,
    min_rating: str = None,
) -> str:
    """Query supplier master data: reliability ratings, lead times, defect rates."""
    try:
        df, err = _load_suppliers()
        if err:
            return err
        if supplier_id:
            df = df[df["supplier_id"].str.strip().str.lower() == supplier_id.strip().lower()].copy()
        if region:
            df = df[df["region"].str.strip().str.lower() == region.strip().lower()].copy()
        if category:
            df = df[df["category"].str.strip().str.lower().str.contains(category.strip().lower(), na=False)].copy()
        if min_rating:
            order = {"D": 0, "C": 1, "B": 2, "A": 3, "A+": 4}
            min_v = order.get(min_rating.strip().upper(), 0)
            df    = df[df["reliability_rating"].map(
                lambda r: order.get(str(r).strip().upper(), 0) >= min_v
            )].copy()
        if df.empty:
            return json.dumps({"error": "No suppliers match the given filters."})
        return json.dumps({
            "total_suppliers":  len(df),
            "avg_lead_time":    round(float(df["avg_lead_time_days"].mean()), 1),
            "avg_on_time_pct":  round(float(df["on_time_delivery_pct"].mean()), 1),
            "avg_defect_rate":  round(float(df["defect_rate_pct"].mean()), 2),
            "suppliers": df[[
                "supplier_id", "supplier_name", "region", "category",
                "reliability_rating", "avg_lead_time_days",
                "on_time_delivery_pct", "defect_rate_pct", "active",
            ]].to_dict(orient="records"),
        })
    except Exception as exc:
        return json.dumps({"error": f"get_supplier_info failed: {exc}"})


# =============================================================================
# SHARED ESCALATION TOOL
# =============================================================================

_ESCALATE_SENTINEL = "__ESCALATE_TO_ROUTER__"


@tool
def escalate_to_router(reason: str) -> str:
    """
    Escalate this request to the Router so it can redirect to the right persona.
    Call this when the question is outside your own domain.
    Args:
        reason: Brief one-sentence explanation of why you are escalating.
    """
    return json.dumps({
        "escalate":  True,
        "reason":    reason,
        "_sentinel": _ESCALATE_SENTINEL,
    })


# =============================================================================
# CHRIS TOOLS
# =============================================================================

@tool
def forecast_demand(product_id: str, region: str = None, periods: int = 3) -> str:
    """
    Forecast future demand using linear regression on historical sales data.
    Projects N months forward from the last available period (max 12).
    """
    try:
        df, err = _load_sales()
        if err:
            return err
        term = product_id.strip().lower()
        mask = (df["product_id"].str.strip().str.lower() == term) | \
               (df["product_name"].str.strip().str.lower().str.contains(term, na=False))
        df = df[mask].copy()
        if df.empty:
            return json.dumps({"error": f"No sales data for '{product_id}'."})
        product_name = df["product_name"].iloc[0]
        if region:
            df = df[df["region"].str.strip().str.lower() == region.strip().lower()].copy()
            if df.empty:
                return json.dumps({"error": f"No data for '{product_id}' in '{region}'."})
        monthly = df.groupby("period").agg(qty=("qty_sold", "sum")).reset_index().sort_values("period").reset_index(drop=True)
        if len(monthly) < 3:
            return json.dumps({"error": "Need at least 3 months of data to forecast."})
        periods = max(1, min(int(periods), 12))
        x = np.arange(len(monthly))
        y = monthly["qty"].values.astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        last_str   = monthly["period"].iloc[-1]
        last_year, last_month = int(last_str[:4]), int(last_str[5:7])
        std_dev    = float(y[-3:].std()) if len(y) >= 3 else float(y.std())
        forecast_data = []
        for i in range(1, periods + 1):
            fm = last_month + i
            fy = last_year + (fm - 1) // 12
            fm = ((fm - 1) % 12) + 1
            fq = max(0.0, slope * (len(monthly) - 1 + i) + intercept)
            forecast_data.append({
                "period":       f"{fy:04d}-{fm:02d}",
                "forecast_qty": round(fq, 0),
                "lower_bound":  round(max(0.0, fq - 1.5 * std_dev), 0),
                "upper_bound":  round(fq + 1.5 * std_dev, 0),
            })
        pct = round(((forecast_data[-1]["forecast_qty"] - float(y[-1])) / float(y[-1])) * 100, 1) if y[-1] else 0
        return json.dumps({
            "product_name":          product_name,
            "region":                region or "All Regions",
            "model":                 "Linear Regression",
            "data_range":            f"{monthly['period'].iloc[0]} to {monthly['period'].iloc[-1]}",
            "historical_months":     len(monthly),
            "monthly_slope":         round(float(slope), 1),
            "trend_direction":       "increasing" if slope > 0 else "decreasing",
            "projected_change_pct":  pct,
            "historical_data":       monthly.to_dict(orient="records"),
            "forecast_data":         forecast_data,
        })
    except Exception as exc:
        return json.dumps({"error": f"forecast_demand failed: {exc}"})


@tool
def calculate_seasonality(product_id: str, region: str = None) -> str:
    """
    Compute monthly seasonal index from 2023-2025 sales history.
    1.0 = average month; >1.0 = above average demand.
    """
    try:
        df, err = _load_sales()
        if err:
            return err
        term = product_id.strip().lower()
        mask = (df["product_id"].str.strip().str.lower() == term) | \
               (df["product_name"].str.strip().str.lower().str.contains(term, na=False))
        df = df[mask].copy()
        if df.empty:
            return json.dumps({"error": f"No sales data for '{product_id}'."})
        product_name = df["product_name"].iloc[0]
        if region:
            df = df[df["region"].str.strip().str.lower() == region.strip().lower()].copy()
            if df.empty:
                return json.dumps({"error": f"No data for '{product_id}' in '{region}'."})
        df["month"]  = df["period"].str[5:7].astype(int)
        monthly_avg  = df.groupby("month")["qty_sold"].mean()
        grand_avg    = monthly_avg.mean()
        if grand_avg == 0:
            return json.dumps({"error": "Cannot compute seasonality: zero average sales."})
        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        factors = [{"month": month_names[m-1], "month_num": m,
                    "seasonal_index": round(float(monthly_avg.get(m, grand_avg)) / grand_avg, 3)}
                   for m in range(1, 13)]
        peak   = max(factors, key=lambda x: x["seasonal_index"])
        trough = min(factors, key=lambda x: x["seasonal_index"])
        return json.dumps({
            "product_name":     product_name,
            "region":           region or "All Regions",
            "peak_month":       peak["month"],
            "peak_index":       peak["seasonal_index"],
            "trough_month":     trough["month"],
            "trough_index":     trough["seasonal_index"],
            "seasonal_factors": factors,
        })
    except Exception as exc:
        return json.dumps({"error": f"calculate_seasonality failed: {exc}"})


@tool
def run_what_if(
    product_id: str,
    region: str = None,
    change_pct: float = 10.0,
    metric: str = "demand",
) -> str:
    """
    Simulate impact of a % change in demand or cost on units/revenue.
    metric: 'demand' or 'cost'. change_pct: positive=increase, negative=decrease.
    """
    try:
        df, err = _load_sales()
        if err:
            return err
        term = product_id.strip().lower()
        mask = (df["product_id"].str.strip().str.lower() == term) | \
               (df["product_name"].str.strip().str.lower().str.contains(term, na=False))
        df = df[mask].copy()
        if df.empty:
            return json.dumps({"error": f"No sales data for '{product_id}'."})
        product_name = df["product_name"].iloc[0]
        if region:
            df = df[df["region"].str.strip().str.lower() == region.strip().lower()].copy()
        last_periods = sorted(df["period"].unique())[-12:]
        base_df      = df[df["period"].isin(last_periods)]
        base_qty     = float(base_df["qty_sold"].sum())
        base_rev     = float(base_df["revenue"].sum())
        mult         = 1 + (change_pct / 100)
        sim_qty = base_qty * mult if metric.lower() == "demand" else base_qty
        sim_rev = base_rev * mult if metric.lower() == "demand" else base_rev
        if metric.lower() not in ("demand", "cost"):
            return json.dumps({"error": "metric must be 'demand' or 'cost'."})
        return json.dumps({
            "product_name":      product_name,
            "region":            region or "All Regions",
            "metric":            metric,
            "change_pct":        change_pct,
            "analysis_period":   f"{last_periods[0]} to {last_periods[-1]}",
            "baseline_qty":      round(base_qty, 0),
            "baseline_revenue":  round(base_rev, 2),
            "simulated_qty":     round(sim_qty, 0),
            "simulated_revenue": round(sim_rev, 2),
            "delta_qty":         round(sim_qty - base_qty, 0),
            "delta_revenue":     round(sim_rev - base_rev, 2),
        })
    except Exception as exc:
        return json.dumps({"error": f"run_what_if failed: {exc}"})


# =============================================================================
# TOOL REGISTRIES
# =============================================================================

BLAKE_TOOLS = [
    get_inventory, get_trend, compare_regions, aggregate,
    get_sales_trend, get_purchase_orders, get_supplier_info,
    escalate_to_router,
]
CHRIS_TOOLS = [forecast_demand, calculate_seasonality, run_what_if, escalate_to_router]

blake_tool_node = ToolNode(BLAKE_TOOLS)
chris_tool_node = ToolNode(CHRIS_TOOLS)


# =============================================================================
# LLM FACTORY
# =============================================================================

def _build_llm(tools=None, temperature: float = 0.1, model_name: str = MODEL_NAME) -> ChatOllama:
    llm = ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=512,
    )
    return llm.bind_tools(tools) if tools else llm


# =============================================================================
# FIX 2: ROBUST ROUTER TOKEN PARSER
# Priority order prevents BLAKE from matching inside ROUTE_TO_CHRIS etc.
# Substring match tolerates extra prose around the token.
# =============================================================================

_CLARIFY_REPLY = (
    "I want to make sure I connect you with the right expert. "
    "Could you clarify — are you looking for historical data, charts, or reports "
    "(Blake's area), or a future forecast or simulation (Chris's domain)?"
)

_OVERFLOW_CLARIFY = (
    "I'm having trouble determining the best person for this request. "
    "Could you clarify what you'd like: historical data/reports (Blake) "
    "or demand forecasting/simulations (Chris)?"
)


def _parse_router_token(text: str) -> str:
    """
    FIX 2: Robust token extraction using substring match with priority ordering.
    Handles models that wrap the token in explanatory prose.
    Priority: CHRIS > BLAKE > CLARIFY > HANDLE (safe fallback).
    """
    upper = text.upper()
    # Longest/most specific first to avoid false substring matches
    if "ROUTE_TO_CHRIS" in upper:
        return "ROUTE_TO_CHRIS"
    if "ROUTE_TO_BLAKE" in upper:
        return "ROUTE_TO_BLAKE"
    if "CLARIFY_USER" in upper:
        return "CLARIFY_USER"
    # FIX 5: Default HANDLE_DIRECTLY only when model clearly intends a greeting/meta.
    # Unknown outputs → CLARIFY, not HANDLE (avoids silently swallowing real queries).
    if any(kw in upper for kw in ("HANDLE_DIRECTLY", "HANDLE DIRECTLY", "HELLO", "HI")):
        return "HANDLE_DIRECTLY"
    return "CLARIFY_USER"  # FIX 5: unknown → ask user, not guess


# =============================================================================
# GRAPH NODES
# =============================================================================

def route_entry(state: AgentState) -> str:
    """Conditional entry: skip router if persona is already active."""
    persona = state.get("active_persona")
    if persona == "blake":
        return "call_blake"
    if persona == "chris":
        return "call_chris"
    return "call_router"


def call_router(state: AgentState) -> AgentState:
    """
    Lightweight router: outputs a routing token, then either handles directly
    or sets active_persona for downstream routing.
    FIX 2: uses _parse_router_token (substring, not equality).
    FIX 5: HANDLE_DIRECTLY strictly for greetings/meta; ambiguous → CLARIFY_USER.
    """
    messages   = state["messages"]
    llm        = _build_llm(temperature=0.0, model_name=ROUTER_MODEL)  # deterministic, no tools
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), messages[-1]
    )
    router_input = [SystemMessage(content=ROUTER_SYSTEM_PROMPT), last_human]
    response     = llm.invoke(router_input)
    token        = _parse_router_token(response.content)

    # Extract any inline reply text after the first line
    lines        = response.content.strip().split("\n")
    inline_reply = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    if token == "ROUTE_TO_BLAKE":
        return {"messages": [], "active_persona": "blake", "escalation_count": 0}

    if token == "ROUTE_TO_CHRIS":
        return {"messages": [], "active_persona": "chris", "escalation_count": 0}

    if token == "CLARIFY_USER":  # FIX 5
        return {
            "messages":       [AIMessage(content=_CLARIFY_REPLY)],
            "active_persona": None,
            "escalation_count": 0,
        }

    # HANDLE_DIRECTLY — router answers for greetings/meta
    reply = inline_reply or (
        "Hello! I'm your Supply Chain AI concierge. "
        "Blake handles data analytics and historical insights. "
        "Chris specialises in demand forecasting and simulations. "
        "What would you like to explore?"
    )
    return {
        "messages":       [AIMessage(content=reply)],
        "active_persona": None,
        "escalation_count": 0,
    }


def after_router(state: AgentState) -> str:
    """Route after router node based on persona set."""
    persona = state.get("active_persona")
    if persona == "blake":
        return "call_blake"
    if persona == "chris":
        return "call_chris"
    return END


def _clean_messages_for_persona(messages: list, system_prompt: str) -> list:
    """
    FIX 6: Build a clean message list for a persona node.
    - Strips any prior SystemMessages (prevent voice bleed from other persona).
    - Re-injects this persona's own SystemMessage at the front.
    - Keeps only HumanMessage and AIMessage pairs from history (strips ToolMessages
      which would confuse a persona that doesn't own those tools).
    """
    clean = [
        m for m in messages
        if isinstance(m, (HumanMessage, AIMessage))
    ]
    return [SystemMessage(content=system_prompt)] + clean


def call_blake(state: AgentState) -> AgentState:
    """Blake node: data analytics LLM with its own isolated context."""
    messages = _clean_messages_for_persona(state["messages"], BLAKE_SYSTEM_PROMPT)
    llm      = _build_llm(tools=BLAKE_TOOLS)
    # Re-attach tool messages needed for multi-step tool loops within this call
    full_messages = [SystemMessage(content=BLAKE_SYSTEM_PROMPT)] + [
        m for m in state["messages"]
        if isinstance(m, (HumanMessage, AIMessage, ToolMessage))
    ]
    # But only if no system message was already there (avoid double injection)
    response = llm.invoke(full_messages)
    return {"messages": [response]}


def call_chris(state: AgentState) -> AgentState:
    """Chris node: forecasting LLM with its own isolated context."""
    full_messages = [SystemMessage(content=CHRIS_SYSTEM_PROMPT)] + [
        m for m in state["messages"]
        if isinstance(m, (HumanMessage, AIMessage, ToolMessage))
    ]
    llm      = _build_llm(tools=CHRIS_TOOLS, model_name=CHRIS_MODEL)
    response = llm.invoke(full_messages)
    return {"messages": [response]}


def should_continue_blake(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        for tc in last.tool_calls:
            if tc["name"] == "escalate_to_router":
                return "escalate"
        return "execute_blake_tools"
    return END


def should_continue_chris(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        for tc in last.tool_calls:
            if tc["name"] == "escalate_to_router":
                return "escalate"
        return "execute_chris_tools"
    return END


def handle_escalation(state: AgentState) -> AgentState:
    """
    FIX 3: Escalation loop guard.
    Increments escalation_count; if >= 2, forces a clarifying response
    instead of bouncing again, preventing infinite ping-pong.
    """
    count = state.get("escalation_count", 0) + 1
    if count >= 2:
        return {
            "messages":       [AIMessage(content=_OVERFLOW_CLARIFY)],
            "active_persona": None,
            "escalation_count": 0,
        }
    
    # The last message was an AIMessage with a tool call to escalate_to_router.
    # We MUST append a ToolMessage to close the tool call, otherwise the LLM API
    # will hang or throw an error when the next persona tries to generate a response
    # with a dangling tool call in its history.
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        tool_call_id = last.tool_calls[0]["id"]
        tool_msg = ToolMessage(
            content=json.dumps({"status": "Escalated to router. Other persona will handle."}),
            name="escalate_to_router",
            tool_call_id=tool_call_id,
        )
        return {"messages": [tool_msg], "active_persona": None, "escalation_count": count}
        
    return {"active_persona": None, "escalation_count": count}


def after_escalation(state: AgentState) -> str:
    """After escalation: check if overflow produced a final reply, else re-route."""
    last = state["messages"][-1] if state["messages"] else None
    # If overflow reply was added (AIMessage with clarification), go to END
    if isinstance(last, AIMessage) and last.content == _OVERFLOW_CLARIFY:
        return END
    return "call_router"


# =============================================================================
# GRAPH ASSEMBLY
# =============================================================================

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("call_router",         call_router)
    builder.add_node("call_blake",          call_blake)
    builder.add_node("call_chris",          call_chris)
    builder.add_node("execute_blake_tools", blake_tool_node)
    builder.add_node("execute_chris_tools", chris_tool_node)
    builder.add_node("handle_escalation",   handle_escalation)

    builder.set_conditional_entry_point(route_entry, {
        "call_router": "call_router",
        "call_blake":  "call_blake",
        "call_chris":  "call_chris",
    })

    builder.add_conditional_edges("call_router", after_router, {
        "call_blake": "call_blake",
        "call_chris": "call_chris",
        END: END,
    })

    builder.add_conditional_edges("call_blake", should_continue_blake, {
        "execute_blake_tools": "execute_blake_tools",
        "escalate":            "handle_escalation",
        END: END,
    })
    builder.add_edge("execute_blake_tools", "call_blake")

    builder.add_conditional_edges("call_chris", should_continue_chris, {
        "execute_chris_tools": "execute_chris_tools",
        "escalate":            "handle_escalation",
        END: END,
    })
    builder.add_edge("execute_chris_tools", "call_chris")

    # FIX 3: escalation goes to its overflow-aware after_escalation, not directly
    builder.add_conditional_edges("handle_escalation", after_escalation, {
        "call_router": "call_router",
        END: END,
    })

    return builder.compile()


# Public compiled graph
graph = build_graph()
