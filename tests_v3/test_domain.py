"""Cart invariants — the pure domain copied into agent_v3 (smoke coverage).

The domain modules are byte-for-byte ports of agent_v2's; this locks in
that the copy + the JSON round-trip used for session_state still hold the
step machine + blockers invariants.
"""

from __future__ import annotations

from agent_v3.checkout import Cart, CartService
from agent_v3.checkout.cart import CheckoutStep
from agent_v3.state import load_cart, save_cart


def test_step_progression():
    svc = CartService()
    assert svc.cart.step == CheckoutStep.COLLECTING_PRODUCTS
    svc.add_item("P-1")
    assert svc.cart.step == CheckoutStep.COLLECTING_IDENTITY
    svc.set_customer("J", "D")
    assert svc.cart.step == CheckoutStep.COLLECTING_ADDRESS


def test_full_flow_to_confirm():
    svc = CartService()
    svc.add_item("P-1")
    svc.set_customer("J", "D")
    svc.set_address("1 Market", "SF", "94110", state="CA")
    svc.lookup_serviceability()
    svc.set_delivery_option("2h")
    svc.quote_shipping()
    svc.compute_tax()
    svc.attach_payment("card", card_token="tok_x")
    assert svc.cart.ready_to_confirm()
    result = svc.confirm()
    assert svc.cart.confirmed
    assert svc.cart.receipt_id
    assert "RCPT" in result or svc.cart.receipt_id in result


def test_blockers_on_empty_cart():
    cart = Cart()
    codes = {b.code for b in cart.blockers()}
    assert "empty_cart" in codes
    assert not cart.ready_to_confirm()


def test_session_state_json_roundtrip_preserves_invariants():
    svc = CartService()
    svc.add_item("P-1", 2)
    svc.set_customer("J", "D")
    ss = {"cart": svc.cart.model_dump(mode="json")}
    cart2 = load_cart(ss)
    assert cart2.step == CheckoutStep.COLLECTING_ADDRESS
    assert str(cart2.subtotal) == str(svc.cart.subtotal)
    # and re-saving is stable
    save_cart(ss, cart2)
    assert load_cart(ss).items[0].quantity == 2
