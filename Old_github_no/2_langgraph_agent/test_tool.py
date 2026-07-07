import sys, json
sys.path.insert(0, r'd:\Work_Dir\Projects\SAP_IBP\2_langgraph_agent')
from agent import get_inventory

# Test 1: Product A, North America
result = get_inventory.invoke({'product_id': 'Product A', 'region': 'North America'})
data = json.loads(result)
assert 'error' not in data, f"Unexpected error: {data}"
print("Test 1 PASS — Product A, North America")
print(f"  product  : {data['product_name']}")
print(f"  trend    : {data['trend_direction']} {data['pct_change']}%")
print(f"  range    : {data['period_range']}")
print(f"  first_qty: {data['first_qty']}  last_qty: {data['last_qty']}")

# Test 2: Product B, all regions (no region filter)
result2 = get_inventory.invoke({'product_id': 'PROD-002'})
data2 = json.loads(result2)
assert 'error' not in data2
assert data2['trend_direction'] == 'decreased'
print("\nTest 2 PASS — Product B (via product_id), all regions")
print(f"  trend    : {data2['trend_direction']} {data2['pct_change']}%")

# Test 3: Not-found guard
result3 = get_inventory.invoke({'product_id': 'Widget XYZ'})
data3 = json.loads(result3)
assert 'error' in data3
print("\nTest 3 PASS — not-found guard")
print(f"  error msg: {data3['error'][:80]}...")

# Test 4: Invalid region guard
result4 = get_inventory.invoke({'product_id': 'Product C', 'region': 'Asia Pacific'})
data4 = json.loads(result4)
assert 'error' in data4
print("\nTest 4 PASS — invalid region guard")
print(f"  error msg: {data4['error'][:80]}...")

print("\nAll tool unit tests passed.")
