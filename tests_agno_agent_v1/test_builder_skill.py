"""AgentSpec → Agent construction + the checkout skill's live anchor injection."""

from __future__ import annotations

from types import SimpleNamespace

from agno_agent_v1.agent.agents import CHECKOUT_SPEC, build_orchestrator, subagent_tools
from agno_agent_v1.agent.builder import build_agent
from agno_agent_v1.agent.context import ShoppingContext
from agno_agent_v1.agent.skills import checkout_anchor_text
from agno_agent_v1.domain import CartService


def test_build_agent_basic():
    agent = build_agent(CHECKOUT_SPEC)
    assert agent.name == "checkout"
    # checkout has its tools + callable (skill) instructions
    tool_names = {getattr(t, "__name__", None) for t in (agent.tools or [])}
    assert "set_customer" in tool_names and "confirm_checkout" in tool_names
    assert callable(agent.instructions)


def test_checkout_skill_instructions_render_live_anchor():
    agent = build_agent(CHECKOUT_SPEC)
    cs = CartService()
    cs.add_item("P-1")
    cs.set_customer("Ada", "Lovelace")
    ctx = ShoppingContext(cart_service=cs)
    # Agno invokes the callable with run_context injected by name
    rendered = agent.instructions(SimpleNamespace(dependencies={"ctx": ctx}))
    assert "Checkout progress" in rendered
    assert "✓ Ada Lovelace" in rendered  # identity already captured
    assert "address" in rendered.lower()


def test_anchor_reflects_repricing_state():
    cs = CartService()
    cs.add_item("P-1")
    cs.set_customer("A", "B")
    cs.set_address("1 Market", "SF", "94105")
    cs.lookup_serviceability()
    cs.set_delivery_option("standard")
    cs.quote_shipping()
    cs.compute_tax()
    text_ready = checkout_anchor_text(cs.cart)
    assert "✓ shipping" in text_ready
    cs.set_quantity("P-1", 4)  # backtrack → stale
    text_stale = checkout_anchor_text(cs.cart)
    assert "STALE" in text_stale


def test_empty_cart_guard_drops_checkout_tool():
    subagent_tools()  # ensure built
    empty = build_orchestrator(cart_empty=True)
    full = build_orchestrator(cart_empty=False)
    assert {t.name for t in empty.tools} == {"product_rec", "order_status"}
    assert {t.name for t in full.tools} == {"product_rec", "checkout", "order_status"}
