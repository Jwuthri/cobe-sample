"""StepResult extraction from Agno tool-execution lists (no LLM)."""

from __future__ import annotations

from agent_agno_v4_1 import extractors as ex
from agent_agno_v4_1 import tools
from tests_agno_v4_1.conftest import tool_exec

_CATALOG_RESULT = (
    "P-1: Classic White T-shirt — $19.99 [apparel, shirt, white]\n"
    "P-2: Black Hoodie — $49.99 [apparel, hoodie, black]"
)


def test_extract_products_parses_catalog_lines():
    out = ex.extract_products([tool_exec("search_products", _CATALOG_RESULT)])
    assert [p["id"] for p in out] == ["P-1", "P-2"]
    assert out[0]["name"] == "Classic White T-shirt"
    assert out[0]["tags"] == ["apparel", "shirt", "white"]


def test_product_rec_catalog_search(ctx):
    sr = ex.extract_product_rec(ctx, [tool_exec("search_products", _CATALOG_RESULT)], before={})
    assert sr.sop == ex.PRODUCT_REC
    assert "catalog returned 2" in sr.summary
    assert sr.details["products"][0]["id"] == "P-1"
    assert sr.asks  # prompts the user to pick


def test_product_rec_add_signals_checkout(run_context, ctx):
    rc = run_context()
    before = {}
    tools.add_item("P-1", rc, quantity=2)  # mutate the real cart
    sr = ex.extract_product_rec(ctx, [tool_exec("add_item", "Added 2 × ...")], before=before)
    assert sr.next_sop == ex.CHECKOUT
    assert sr.details["added"] == ["P-1"]


def test_product_rec_quantity_decrease_is_cart_edit(run_context, ctx):
    rc = run_context()
    tools.add_item("P-1", rc, quantity=2)
    before = {"P-1": 2}
    tools.set_quantity("P-1", 1, rc)
    sr = ex.extract_product_rec(ctx, [tool_exec("set_quantity", "Set P-1 to 1")], before=before)
    assert "updated cart" in sr.summary
    assert sr.details["cart_edit"]["decreased"] == ["P-1"]


def test_checkout_extract_reports_step_and_asks(run_context, ctx):
    rc = run_context()
    tools.add_item("P-1", rc)
    sr = ex.extract_checkout(ctx, [tool_exec("get_cart_summary", "...")], before={})
    assert sr.sop == ex.CHECKOUT
    assert "collecting_identity" in sr.summary
    assert sr.asks == ["first name", "last name"]


def test_order_status_extract():
    raw = "Order ORD-7 is shipped, items=['P-1'], tracking: https://x/ORD-7"
    sr = ex.extract_order_status(None, [tool_exec("get_order_status", raw)], before={})
    assert sr.sop == ex.ORDER_STATUS
    assert sr.summary == "looked up order status"
    assert sr.details["raw"].startswith("Order ORD-7")


def test_order_status_unknown():
    sr = ex.extract_order_status(None, [tool_exec("get_order_status", "unknown order: ORD-9")], {})
    assert "could not find" in sr.summary
    assert sr.asks == ["confirm the order id"]
