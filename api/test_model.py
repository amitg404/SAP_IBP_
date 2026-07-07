"""
test_model.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# ── Config ────────────────────────────────────────────────────────────────────
TEST_MODEL   = os.getenv("BILLY_MODEL",     "gemini-3-flash-preview:cloud")
BASE_URL     = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
API_KEY      = os.getenv("OLLAMA_API_KEY",  "")
TEST_QUERY   = "Has inventory gone up or down for Product A?"

sep = "=" * 65
print(sep)
print("  Billy -- Model Tool-Call Validation Test")
print(sep)
print(f"  Model  : {TEST_MODEL}")
print(f"  URL    : {BASE_URL}")
print(f"  API key: {'SET (' + API_KEY[:8] + '...)' if API_KEY else 'NOT SET'}")
print(sep)


@tool
def get_inventory(product_id: str, region: str = None) -> str:
    """Retrieve inventory data for a product."""
    return json.dumps({"product": product_id, "qty": 100})


try:
    kwargs = dict(
        model=TEST_MODEL,
        base_url=BASE_URL,
        temperature=0.1,
        num_predict=256,
    )
    if API_KEY:
        kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {API_KEY}"}}
        
    llm = ChatOllama(**kwargs).bind_tools([get_inventory])

    print(f"\n  Sending: \"{TEST_QUERY}\"")
    print("  Waiting for response...\n")

    response = llm.invoke([HumanMessage(content=TEST_QUERY)])

    print(f"  Response type : {type(response).__name__}")
    print(f"  Content       : {response.content!r}")

    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        print(f"\n  [PASS] TOOL CALLS DETECTED: {len(tool_calls)}")
        sys.exit(0)
    else:
        print("\n  [WARN] No tool_calls in response")
        sys.exit(1)

except Exception as exc:
    print(f"\n  [ERROR] {exc}\n")
    sys.exit(1)
