"""
Router test suite — FIX 7: comprehensive automated routing validation.

Usage:
  python api/test_router.py

Tests the _parse_router_token function AND (if --live flag) makes real LLM
router calls to validate routing drift across model swaps.
Pass/fail counts reported at end. Re-run whenever you change the underlying model.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from agent import _parse_router_token

# ---------------------------------------------------------------------------
# Static parse tests (no LLM needed) -- FIX 2 robustness validation
# ---------------------------------------------------------------------------
PARSE_CASES = [
    # (raw_model_output, expected_token)
    # Clean tokens
    ("ROUTE_TO_BLAKE",                              "ROUTE_TO_BLAKE"),
    ("ROUTE_TO_CHRIS",                              "ROUTE_TO_CHRIS"),
    ("HANDLE_DIRECTLY",                             "HANDLE_DIRECTLY"),
    ("CLARIFY_USER",                                "CLARIFY_USER"),
    # Wrapped in prose (weak model behaviour)
    ("Based on the query, I think ROUTE_TO_BLAKE.", "ROUTE_TO_BLAKE"),
    ("The answer is ROUTE_TO_CHRIS because...",    "ROUTE_TO_CHRIS"),
    ("I would say CLARIFY_USER here.",             "CLARIFY_USER"),
    ("Hello! HANDLE_DIRECTLY\nHi there!",          "HANDLE_DIRECTLY"),
    # Lowercase tolerance
    ("route_to_blake",                              "ROUTE_TO_BLAKE"),
    ("Route_To_Chris",                              "ROUTE_TO_CHRIS"),
    # Unknown / garbage -> safe fallback
    ("I don't know what to do here.",              "CLARIFY_USER"),
    ("",                                            "CLARIFY_USER"),
    ("42",                                          "CLARIFY_USER"),
    # Priority: CHRIS wins over BLAKE if both appear
    ("ROUTE_TO_CHRIS and also ROUTE_TO_BLAKE",     "ROUTE_TO_CHRIS"),
]

# ---------------------------------------------------------------------------
# Semantic intent cases (for live LLM testing with --live flag)
# Each tuple: (user_message, expected_token)
# ---------------------------------------------------------------------------
SEMANTIC_CASES = [
    # ── HANDLE_DIRECTLY ───────────────────────────────────────────────────
    ("hi",                                              "HANDLE_DIRECTLY"),
    ("hello",                                           "HANDLE_DIRECTLY"),
    ("hey there",                                       "HANDLE_DIRECTLY"),
    ("what can you do?",                                "HANDLE_DIRECTLY"),
    ("who are you?",                                    "HANDLE_DIRECTLY"),
    ("help",                                            "HANDLE_DIRECTLY"),

    # ── ROUTE_TO_BLAKE ────────────────────────────────────────────────────
    ("show inventory for Product A",                    "ROUTE_TO_BLAKE"),
    ("how much stock do we have for Product B?",        "ROUTE_TO_BLAKE"),
    ("what is the inventory trend for Product C?",      "ROUTE_TO_BLAKE"),
    ("compare Product A across regions",                "ROUTE_TO_BLAKE"),
    ("which suppliers have A+ ratings?",                "ROUTE_TO_BLAKE"),
    ("show me purchase orders for Product D",           "ROUTE_TO_BLAKE"),
    ("what are the open POs for supplier SUP-001?",     "ROUTE_TO_BLAKE"),
    ("sales revenue by channel last year",              "ROUTE_TO_BLAKE"),
    ("how did Product E perform in Europe in 2024?",    "ROUTE_TO_BLAKE"),
    ("which region has the highest inventory?",         "ROUTE_TO_BLAKE"),
    ("on-time delivery rate for our suppliers",         "ROUTE_TO_BLAKE"),
    ("show me a chart of sales trends",                 "ROUTE_TO_BLAKE"),
    ("what is the defect rate for our suppliers?",      "ROUTE_TO_BLAKE"),
    ("how much revenue did we make from Product F?",    "ROUTE_TO_BLAKE"),

    # ── ROUTE_TO_CHRIS ───────────────────────────────────────────────────
    ("forecast demand for Product A next quarter",      "ROUTE_TO_CHRIS"),
    ("what will inventory look like in 3 months?",      "ROUTE_TO_CHRIS"),
    ("predict sales for Product B in Europe",           "ROUTE_TO_CHRIS"),
    ("what if demand drops by 20%?",                    "ROUTE_TO_CHRIS"),
    ("run a what-if simulation on Product C costs",     "ROUTE_TO_CHRIS"),
    ("calculate safety stock for Product D",            "ROUTE_TO_CHRIS"),
    ("what is the seasonal pattern for Product E?",     "ROUTE_TO_CHRIS"),
    ("project demand for next 6 months",                "ROUTE_TO_CHRIS"),
    ("how is demand expected to grow next year?",       "ROUTE_TO_CHRIS"),
    ("simulate a 15% increase in sales",                "ROUTE_TO_CHRIS"),

    # ── CLARIFY_USER ─────────────────────────────────────────────────────
    ("tell me about Product A",                         "CLARIFY_USER"),
    ("analyse Product B",                               "CLARIFY_USER"),
    ("help me with Product C planning",                 "CLARIFY_USER"),
    ("what should I do about inventory?",               "CLARIFY_USER"),
]


def run_parse_tests() -> tuple[int, int]:
    passed, failed = 0, 0
    print("\n=== Static Parser Tests ===")
    for raw, expected in PARSE_CASES:
        got = _parse_router_token(raw)
        ok  = got == expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  [{status}] input={repr(raw[:60])} expected={expected} got={got}")
            failed += 1
        else:
            passed += 1
    print(f"  Result: {passed} passed, {failed} failed out of {len(PARSE_CASES)}")
    return passed, failed


def run_live_tests(router_fn) -> tuple[int, int]:
    """
    Runs semantic cases against the real router LLM.
    Pass router_fn = a callable(user_message) -> token string.
    """
    passed, failed = 0, 0
    print("\n=== Live Router LLM Tests ===")
    for msg, expected in SEMANTIC_CASES:
        try:
            got = router_fn(msg)
            ok  = got == expected
        except Exception as exc:
            got, ok = f"ERROR: {exc}", False
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  [{status}] msg={repr(msg[:50])} expected={expected} got={got}")
            failed += 1
        else:
            passed += 1
    print(f"  Result: {passed} passed, {failed} failed out of {len(SEMANTIC_CASES)}")
    return passed, failed


if __name__ == "__main__":
    total_pass, total_fail = run_parse_tests()

    if "--live" in sys.argv:
        from langchain_core.messages import HumanMessage, SystemMessage
        from agent import _build_llm, ROUTER_SYSTEM_PROMPT, _parse_router_token, ROUTER_MODEL  # noqa

        def _live_router(msg: str) -> str:
            llm = _build_llm(temperature=0.0, model_name=ROUTER_MODEL)
            resp = llm.invoke([
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=msg),
            ])
            return _parse_router_token(resp.content)

        p, f = run_live_tests(_live_router)
        total_pass += p
        total_fail += f

    print(f"\n{'='*50}")
    print(f"TOTAL: {total_pass} passed, {total_fail} failed")
    if total_fail > 0:
        print("ACTION: Fix failing cases before deploying a new model.")
    else:
        print("All tests passed.")
    print('='*50)
    sys.exit(0 if total_fail == 0 else 1)
