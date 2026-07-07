import sys, json
sys.path.insert(0, r'd:\Work_Dir\Projects\SAP_IBP\backend')
from agent import graph, get_inventory, MODEL_NAME, CSV_PATH
import pandas as pd

print("=== Backend Import Verification ===")
print(f"Graph nodes    : {list(graph.nodes.keys())}")
print(f"Model          : {MODEL_NAME}")
print(f"CSV path       : {CSV_PATH}")
print(f"CSV exists     : {CSV_PATH.exists()}")

# Tool: successful query
r = get_inventory.invoke({"product_id": "Product A", "region": "North America"})
d = json.loads(r)
assert "error" not in d, f"Tool FAILED: {d}"
print(f"Tool PASS      : {d['trend_direction']} {d['pct_change']}% ({d['period_range']})")

# Tool: product not found guard
r2 = get_inventory.invoke({"product_id": "Widget XYZ"})
d2 = json.loads(r2)
assert "error" in d2, "Not-found guard FAILED"
print(f"Not-found guard: PASS - {d2['error'][:60]}...")

# Tool: invalid region guard
r3 = get_inventory.invoke({"product_id": "Product B", "region": "Asia Pacific"})
d3 = json.loads(r3)
assert "error" in d3
print(f"Bad region guard: PASS - {d3['error'][:60]}...")

# CSV schema validation
df = pd.read_csv(CSV_PATH)
required = {"product_id", "product_name", "region", "period", "inventory_qty"}
missing = required - set(df.columns)
print(f"CSV columns    : {sorted(df.columns.tolist())}")
print(f"Missing cols   : {missing if missing else 'None -- schema OK'}")
print(f"CSV rows       : {len(df)} (expected 96)")
assert len(df) == 96, f"Expected 96 rows, got {len(df)}"

# FastAPI schema test
from main import app, ChatRequest, ChatResponse
from fastapi.testclient import TestClient
client = TestClient(app, raise_server_exceptions=False)

r = client.get("/health")
assert r.status_code == 200
print(f"Health check   : PASS - {r.json()}")

r = client.post("/chat", json={"message": ""})
assert r.status_code == 422
print(f"Empty msg guard: PASS - 422 Unprocessable")

r = client.post("/chat", json={"message": "What is the demand forecast for Product A?"})
assert r.status_code == 200
body = r.json()
assert "response" in body
assert list(body.keys()) == ["response"]
print(f"Scope guard    : PASS - {body['response'][:70]}...")

print()
print("All backend integration checks PASSED.")
