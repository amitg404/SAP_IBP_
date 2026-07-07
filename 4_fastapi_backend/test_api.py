import sys
sys.path.insert(0, r'd:\Work_Dir\Projects\SAP_IBP\4_fastapi_backend')
from main import app, ChatRequest, ChatResponse
from fastapi.testclient import TestClient

client = TestClient(app, raise_server_exceptions=False)

# Test 1: health check
r = client.get('/health')
assert r.status_code == 200, f"Expected 200, got {r.status_code}"
print(f"Test 1 PASS - GET /health: {r.json()}")

# Test 2: schema validation - empty message rejected
r = client.post('/chat', json={'message': ''})
assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"
print("Test 2 PASS - empty message => 422 validation error")

# Test 3: schema validation - unknown field rejected (extra=forbid)
r = client.post('/chat', json={'message': 'hi', 'extra_field': 'bad'})
assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"
print("Test 3 PASS - extra field => 422 validation error")

# Test 4: out-of-scope query pre-flight guard fires (no LLM call needed)
r = client.post('/chat', json={'message': 'What is the demand forecast for Product A?'})
assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
data = r.json()
assert 'response' in data, "Missing 'response' key"
resp_lower = data['response'].lower()
assert 'inventory' in resp_lower or 'optimised' in resp_lower or 'optimized' in resp_lower, \
    f"Expected refusal message, got: {data['response']}"
print(f"Test 4 PASS - out-of-scope => pre-flight refusal (first 80 chars): {data['response'][:80]}")

# Test 5: missing payload field
r = client.post('/chat', json={})
assert r.status_code == 422
print("Test 5 PASS - missing 'message' field => 422 validation error")

# Test 6: response model shape
r = client.post('/chat', json={'message': 'What is the forecast for Product B?'})
assert r.status_code == 200
data = r.json()
assert list(data.keys()) == ['response'], f"Unexpected keys in response: {data.keys()}"
print(f"Test 6 PASS - response shape correct: keys={list(data.keys())}")

print()
print("All FastAPI schema + routing tests passed.")
