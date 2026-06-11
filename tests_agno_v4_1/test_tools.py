"""Tools mutate the shared cart through the injected RunContext dependencies."""

from __future__ import annotations

from agent_agno_v4_1 import tools
from agent_agno_v4_1.context import ShoppingContext


def test_add_item_mutates_shared_cart(run_context, ctx):
    rc = run_context()
    msg = tools.add_item("P-1", rc, quantity=2)
    assert "Added 2" in msg
    assert [(i.product_id, i.quantity) for i in ctx.cart_service.cart.items] == [("P-1", 2)]


def test_set_address_tolerates_none_country(run_context, ctx):
    # The model sometimes passes country=None for an optional-with-default str.
    rc = run_context()
    tools.add_item("P-1", rc)
    msg = tools.set_address("1 Main", "SF", "94110", rc, country=None)
    assert "Address set" in msg
    assert ctx.cart_service.cart.address.country == "US"


def test_confirm_checkout_gated_by_blockers(run_context, ctx):
    rc = run_context()
    tools.add_item("P-1", rc)
    out = tools.confirm_checkout(rc)
    assert out.startswith("error: cannot confirm")  # missing identity/address/etc.


def test_full_checkout_confirms_and_persists_to_store(run_context):
    ctx = ShoppingContext()
    from agent_v4_1.shopping.domain.memory import build_store, recent_orders

    ctx.store = build_store()
    rc = run_context(ctx)
    tools.add_item("P-1", rc, quantity=2)
    tools.set_customer("Julien", "Wuthrich", rc)
    tools.set_address("1717 Webster", "San Francisco", "94110", rc)
    tools.lookup_serviceability(rc)
    tools.set_delivery_option("standard", rc)
    tools.quote_shipping(rc)
    tools.compute_tax(rc)
    tools.attach_payment("cash", rc)
    out = tools.confirm_checkout(rc)
    assert ctx.cart_service.cart.confirmed
    assert ctx.cart_service.cart.receipt_id == "RCPT-9000"
    assert "RCPT-9000" in out
    assert recent_orders(ctx.store, ctx.user_id)  # persisted
