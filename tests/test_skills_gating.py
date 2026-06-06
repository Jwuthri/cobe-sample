"""Constrained tools refuse until their required skill is loaded."""

from __future__ import annotations

from agent_v2.checkout import CartService
from agent_v2.runtime import RuntimeContext
from agent_v2.tools.checkout_tools import (
    add_item,
    confirm_checkout,
    lookup_serviceability,
    set_address,
    set_customer,
)
from langchain.tools import ToolRuntime


def _runtime(state: dict | None, service: CartService) -> ToolRuntime:
    return ToolRuntime(
        state=state,
        context=RuntimeContext(user_id="u", session_id="s", cart_service=service),
        config={},
        stream_writer=lambda _: None,
        tool_call_id="test",
        store=None,
        tools=[],
    )


def test_set_customer_refuses_without_skill():
    s = CartService()
    result = set_customer.invoke(
        {
            "first_name": "A",
            "last_name": "B",
            "runtime": _runtime({"skills_loaded": []}, s),
        }
    )
    assert "collect_identity" in result
    assert s.cart.customer.first_name is None


def test_set_customer_works_with_skill_loaded():
    s = CartService()
    result = set_customer.invoke(
        {
            "first_name": "A",
            "last_name": "B",
            "runtime": _runtime({"skills_loaded": ["collect_identity"]}, s),
        }
    )
    assert "Customer set to A B" in result
    assert s.cart.customer.first_name == "A"


def test_set_address_refuses_without_skill():
    s = CartService()
    result = set_address.invoke(
        {
            "street": "x",
            "city": "y",
            "zip_code": "94110",
            "runtime": _runtime({"skills_loaded": ["collect_identity"]}, s),
        }
    )
    assert "collect_address" in result


def test_lookup_serviceability_refuses_without_skill():
    s = CartService()
    s.set_address("x", "y", "94110")  # via service, not via @tool — bypass guard
    result = lookup_serviceability.invoke({"runtime": _runtime({"skills_loaded": []}, s)})
    assert "lookup_serviceability" in result


def test_confirm_refuses_when_not_ready_even_with_skill():
    s = CartService()
    # Skill loaded, but cart isn't ready_to_confirm.
    result = confirm_checkout.invoke(
        {"runtime": _runtime({"skills_loaded": ["collect_payment"]}, s)}
    )
    assert "blockers" in result
    assert not s.cart.confirmed


def test_add_item_always_callable():
    """add_item has no skill gate — products are step 0."""
    s = CartService()
    result = add_item.invoke(
        {
            "product_id": "P-2",
            "runtime": _runtime({"skills_loaded": []}, s),
        }
    )
    assert "Added" in result
    assert len(s.cart.items) == 1
