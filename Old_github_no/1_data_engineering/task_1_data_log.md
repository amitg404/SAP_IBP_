# Task 1 — Data Engineering Log

**Mock SAP IBP Data Layer**  
Simulates: `/IBP/PLANNING_DATA_API_SRV`  
File: `inventory.csv` | Encoding: UTF-8 | Separator: `,`

---

## 1. Schema Definition

| Column | Data Type | Format / Example | Notes |
|---|---|---|---|
| `product_id` | String (Alphanumeric) | `PROD-001` | Unique SKU identifier |
| `product_name` | String | `Product A` | Human-readable label |
| `region` | String | `North America`, `Europe` | Geo filter dimension |
| `period` | String (ISO 8601 YYYY-MM) | `2025-01` | Chronological month key |
| `inventory_qty` | Integer | `10200` | Stock count; no decimals |

---

## 2. SKUs Generated & Injected Trends

### Coverage
- **Products:** 4 SKUs (`PROD-001` to `PROD-004`)
- **Regions:** 2 (`North America`, `Europe`)
- **Periods:** 12 months (`2025-01` → `2025-12`)
- **Total rows:** 96

### Trend Summary

| Product | product_id | Trend Type | North America Start→End | North America Δ% | Europe Start→End | Europe Δ% |
|---|---|---|---|---|---|---|
| Product A | PROD-001 | 📈 Upward | 10,200 → 13,500 | **+32.4%** | 8,100 → 10,700 | **+32.1%** |
| Product B | PROD-002 | 📉 Downward | 15,000 → 9,100 | **−39.3%** | 12,000 → 7,300 | **−39.2%** |
| Product C | PROD-003 | ➡️ Flat/Stable | 7,800 → 8,050 | **+3.2%** | 6,100 → 6,100 | **0.0%** |
| Product D | PROD-004 | 〰️ Volatile/Seasonal | 5,000 → 7,900 | High variance | 4,200 → 6,500 | High variance |

### Trend Details

**Product A (PROD-001) — Upward:**  
Steady monthly increase of ~250–350 units. Clear, unambiguous growth story.  
Demo question → *"Has inventory gone up or down over the last 6 months for Product A?"*  
Expected answer → Inventory has risen approximately **32%** over 12 months.

**Product B (PROD-002) — Downward:**  
Steady monthly depletion of ~400–500 units. Simulates stock drawdown / supply chain issue.  
Demo question → *"Is Product B running low?"*  
Expected answer → Inventory has dropped approximately **39%** since January 2025.

**Product C (PROD-003) — Flat/Stable:**  
Random walk ±150 units around baseline (7,900 NA / 6,150 EU). Net change <5%.  
Demo question → *"Is Product C inventory stable?"*  
Expected answer → Inventory has remained essentially flat with minor fluctuations.

**Product D (PROD-004) — Volatile/Seasonal:**  
Alternating high-low pattern simulating demand spikes or batch replenishment cycles.  
Useful for testing the LLM's ability to describe non-linear trends.

---

## 3. Parsing Instructions for Task 2 (LangGraph Agent)

### `period` Column

- **Format:** `YYYY-MM` string (e.g., `"2025-01"`)
- **Sorting:** Standard lexicographic string sort (`sorted(periods)`) is safe and correct because the format is zero-padded ISO 8601. No `datetime` parsing required, though it is acceptable.

```python
import pandas as pd

df = pd.read_csv("inventory.csv")

# Filter by product and optional region
filtered = df[df["product_name"] == "Product A"]
# optional: filtered = filtered[filtered["region"] == "North America"]

# Sort chronologically — string sort works on YYYY-MM
filtered = filtered.sort_values("period")

# Aggregate across regions if no region filter
monthly = filtered.groupby("period")["inventory_qty"].sum().reset_index()

# Trend calculation
first_qty = monthly.iloc[0]["inventory_qty"]
last_qty  = monthly.iloc[-1]["inventory_qty"]
pct_change = ((last_qty - first_qty) / first_qty) * 100
```

### Key Parsing Rules
1. **Sort before any calculation.** Never assume CSV row order is chronological.
2. **Group by `period` then sum `inventory_qty`** when no region filter is given — otherwise you'll double-count.
3. **Percentage change formula:** `((last - first) / first) * 100`. Round to 1 decimal place in the final LLM response.
4. **Case-insensitive matching:** User may type `"product a"` or `"PRODUCT A"`. Normalise with `.str.lower().str.strip()` before filtering.
5. **Not-found guard:** If `filtered.empty` after applying filters, return `"No data found for [product_name]"` — do NOT let the LLM hallucinate a number.

---

## 4. Architecture Connection

```
inventory.csv  ──►  LangGraph Tool (Task 2)  ──►  Pandas filter + sort + aggregate
                                                      │
                                                      ▼
                                             FastAPI POST /chat (Task 4)
                                                      │
                                                      ▼
                                             React UI response (Task 5)
```

The CSV is the **single source of truth** for all inventory queries in the MVP.  
It is read at tool-call time (no in-memory cache) — acceptable for POC scale (<200 rows).

---

*Generated: 2026-07-07 | Billy MVP — Task 1 Complete*
