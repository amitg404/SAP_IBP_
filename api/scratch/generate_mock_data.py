"""
generate_mock_data.py — creates three new SAP IBP-style CSV datasets.

Files generated:
  backend/sales_history.csv     — 1200+ rows, multi-product/region/month
  backend/purchase_orders.csv   — 480 rows, supplier POs
  backend/suppliers.csv         — 60 rows, supplier master data

Run once:
  cd backend
  python scratch/generate_mock_data.py
"""

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

OUT = Path(__file__).parent.parent  # backend/

# ── Shared dimensions ─────────────────────────────────────────────────────────
PRODUCTS = [
    ("PROD-001", "Product A"),
    ("PROD-002", "Product B"),
    ("PROD-003", "Product C"),
    ("PROD-004", "Product D"),
    ("PROD-005", "Product E"),
    ("PROD-006", "Product F"),
    ("PROD-007", "Product G"),
    ("PROD-008", "Product H"),
    ("PROD-009", "Product I"),
    ("PROD-010", "Product J"),
]

REGIONS = [
    "North America",
    "Europe",
    "Asia Pacific",
    "Middle East & Africa",
    "Latin America",
]

CHANNELS = ["Direct", "Retail", "Online", "Distributor", "Wholesale"]

SUPPLIERS = [
    ("SUP-001", "Apex Manufacturing", "North America"),
    ("SUP-002", "Euro Components GmbH", "Europe"),
    ("SUP-003", "Pacific Parts Co.", "Asia Pacific"),
    ("SUP-004", "Sahara Supplies", "Middle East & Africa"),
    ("SUP-005", "AmeriGoods SA", "Latin America"),
    ("SUP-006", "Nordic Industrial", "Europe"),
    ("SUP-007", "Eastgate Logistics", "Asia Pacific"),
    ("SUP-008", "Atlas Distribution", "North America"),
    ("SUP-009", "Delta Components", "Europe"),
    ("SUP-010", "Pacific Rim Trading", "Asia Pacific"),
]

# Months: 2023-01 → 2025-12 (36 months)
def months():
    d = date(2023, 1, 1)
    while d <= date(2025, 12, 1):
        yield d.strftime("%Y-%m")
        d = date(d.year + (d.month // 12), (d.month % 12) + 1, 1)

PERIODS = list(months())


# ── 1. sales_history.csv  (1200 rows) ─────────────────────────────────────────
def gen_sales():
    path = OUT / "sales_history.csv"
    rows = []
    for prod_id, prod_name in PRODUCTS:
        for region in REGIONS:
            # Base sales volume varies by product and region
            base = random.randint(4_000, 18_000)
            growth = random.uniform(0.005, 0.025)          # monthly growth
            seasonality = [1.0, 0.95, 1.05, 1.1, 1.08, 1.15,
                           1.2, 1.18, 1.12, 1.25, 1.35, 1.40]  # Jan-Dec
            for i, period in enumerate(PERIODS):
                month_idx = int(period[-2:]) - 1
                trend_factor = (1 + growth) ** i
                seasonal_factor = seasonality[month_idx]
                noise = random.uniform(0.92, 1.08)
                qty = max(100, int(base * trend_factor * seasonal_factor * noise))
                unit_price = round(random.uniform(12.5, 89.9), 2)
                channel = random.choice(CHANNELS)
                rows.append({
                    "order_id":    f"ORD-{len(rows)+1:06d}",
                    "product_id":  prod_id,
                    "product_name": prod_name,
                    "region":      region,
                    "channel":     channel,
                    "period":      period,
                    "qty_sold":    qty,
                    "unit_price":  unit_price,
                    "revenue":     round(qty * unit_price, 2),
                    "returns_qty": random.randint(0, max(1, qty // 20)),
                })
    random.shuffle(rows)
    print(f"  sales_history.csv: {len(rows)} rows")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


# ── 2. purchase_orders.csv  (480 rows) ────────────────────────────────────────
PO_STATUSES = ["Open", "In Transit", "Received", "Partial", "Cancelled"]

def gen_purchase_orders():
    path = OUT / "purchase_orders.csv"
    rows = []
    po_num = 1
    for prod_id, prod_name in PRODUCTS:
        for sup_id, sup_name, sup_region in SUPPLIERS[:6]:  # 6 suppliers × 10 products × 8 orders
            for _ in range(8):
                period = random.choice(PERIODS)
                due_offset = random.randint(7, 60)
                due = date(int(period[:4]), int(period[-2:]), 1) + timedelta(days=due_offset)
                order_qty = random.randint(500, 10_000)
                unit_cost = round(random.uniform(5.0, 45.0), 2)
                received_qty = (
                    0 if "Open" in PO_STATUSES or random.random() < 0.05
                    else int(order_qty * random.uniform(0.7, 1.0))
                )
                rows.append({
                    "po_number":    f"PO-{po_num:05d}",
                    "product_id":   prod_id,
                    "product_name": prod_name,
                    "supplier_id":  sup_id,
                    "supplier_name": sup_name,
                    "region":       sup_region,
                    "period":       period,
                    "due_date":     due.strftime("%Y-%m-%d"),
                    "order_qty":    order_qty,
                    "received_qty": received_qty,
                    "unit_cost":    unit_cost,
                    "total_cost":   round(order_qty * unit_cost, 2),
                    "status":       random.choice(PO_STATUSES),
                    "lead_time_days": random.randint(5, 45),
                })
                po_num += 1
    print(f"  purchase_orders.csv: {len(rows)} rows")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


# ── 3. suppliers.csv  (60 rows — 10 suppliers × 6 product categories) ────────
CATEGORIES = ["Electronics", "Raw Materials", "Packaging", "Chemicals", "Machinery", "Consumables"]
RATINGS = ["A+", "A", "B+", "B", "C"]

def gen_suppliers():
    path = OUT / "suppliers.csv"
    rows = []
    for sup_id, sup_name, sup_region in SUPPLIERS:
        for cat in CATEGORIES:
            rows.append({
                "supplier_id":         sup_id,
                "supplier_name":       sup_name,
                "region":              sup_region,
                "category":            cat,
                "reliability_rating":  random.choice(RATINGS),
                "avg_lead_time_days":  random.randint(5, 60),
                "min_order_qty":       random.choice([100, 250, 500, 1000]),
                "unit_cost_usd":       round(random.uniform(3.0, 50.0), 2),
                "active":              random.choice([True, True, True, False]),
                "contract_expiry":     f"202{random.randint(6,9)}-{random.randint(1,12):02d}-01",
                "on_time_delivery_pct": round(random.uniform(72.0, 99.5), 1),
                "defect_rate_pct":     round(random.uniform(0.1, 4.5), 2),
            })
    print(f"  suppliers.csv: {len(rows)} rows")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    print("Generating mock SAP IBP datasets...")
    gen_sales()
    gen_purchase_orders()
    gen_suppliers()
    print("Done.")
