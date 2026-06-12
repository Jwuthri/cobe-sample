"""The five agents, declared as data, plus the builders that compile them.

  * three sub-agents (product_rec / checkout / order_status), each an
    :class:`AgentSpec` + a :class:`SubagentSpec` wiring its domain hooks;
  * the writer (a no-tools agent — the user-facing streamed voice);
  * the orchestrator (routes to the sub-agent tools, then emits DONE).

The orchestrator is built *per turn* so the empty-cart guard can drop the
checkout tool while the cart is empty — making an "add X" → checkout misroute
structurally impossible (agent_v4_1's ``empty_cart_guard``, ported).
"""

from __future__ import annotations

from typing import Any

from agno_agent_v1.agent.builder import AgentSpec, build_agent
from agno_agent_v1.agent.extractors import (
    cart_quantities,
    checkout_input,
    extract_checkout,
    extract_order_status,
    extract_product_rec,
    summarize_checkout,
    summarize_product_rec,
    with_cart_note,
)
from agno_agent_v1.agent.prompts import (
    CHECKOUT_PROMPT,
    ORDER_STATUS_PROMPT,
    PRODUCT_REC_PROMPT,
    ROUTER_PROMPT,
    WRITER_SYSTEM,
)
from agno_agent_v1.agent.skills import CHECKOUT_PROGRESS
from agno_agent_v1.agent.subagent import SubagentSpec, build_subagent_tools
from agno_agent_v1.agent.tools import CHECKOUT_TOOLS, ORDER_STATUS_TOOLS, PRODUCT_REC_TOOLS

# Temperatures mirror agent_v4_1 (router/sub-agents deterministic, writer warmer).
# resolve_model drops the value for gpt-5.x / o-series, which reject a custom one.
_DETERMINISTIC = 0.0
_WRITER_TEMP = 0.3


# ===========================================================================
# agent specs
# ===========================================================================
PRODUCT_REC_SPEC = AgentSpec(
    name="product_rec",
    description="Browse + cart management: search, lookup, serviceability, add/remove/qty/view.",
    prompt=PRODUCT_REC_PROMPT,
    tools=PRODUCT_REC_TOOLS,
    temperature=_DETERMINISTIC,
)

CHECKOUT_SPEC = AgentSpec(
    name="checkout",
    description="Drive an in-progress purchase from identity to payment to confirmation.",
    prompt=CHECKOUT_PROMPT,
    tools=CHECKOUT_TOOLS,
    skills=[CHECKOUT_PROGRESS],  # injects the live "Checkout progress" anchor
    temperature=_DETERMINISTIC,
)

ORDER_STATUS_SPEC = AgentSpec(
    name="order_status",
    description="Look up a past order's status / tracking.",
    prompt=ORDER_STATUS_PROMPT,
    tools=ORDER_STATUS_TOOLS,
    temperature=_DETERMINISTIC,
)

WRITER_SPEC = AgentSpec(
    name="writer",
    description="Compose the single user-facing reply from verified step results + cart.",
    prompt=WRITER_SYSTEM,
    tools=[],
    temperature=_WRITER_TEMP,
)

ORCHESTRATOR_SPEC = AgentSpec(
    name="orchestrator",
    description="Route the user's message to its sub-agents, then emit DONE.",
    prompt=ROUTER_PROMPT,
    tools=[],  # the delegates are the sub-agent tools, injected at build time
    temperature=_DETERMINISTIC,
    tool_call_limit=4,  # backstop against a sub-agent-call loop
)


# ===========================================================================
# sub-agent specs (config + domain hooks) — wrapped as the orchestrator's tools
# ===========================================================================
_PRODUCT_REC_DESC = (
    "Search the catalog, look up a product, answer delivery-area questions, and "
    "edit the cart (add / remove / change quantity / show contents). Call this for "
    "ANY browsing or cart-content request. Pass a self-contained instruction as "
    "`query` (e.g. 'add P-2 to the cart', 'search for caps'). Adding an item is a "
    "natural cue to proceed to checkout next."
)
_CHECKOUT_DESC = (
    "Move an order forward: capture identity, shipping address, delivery option, "
    "and payment, then place the order ONLY on the user's explicit 'yes'. Requires "
    "items already in the cart. Pass the user's latest checkout-relevant message as "
    "`query` (their name, an address, a delivery choice, a payment method, or 'yes')."
)
_ORDER_STATUS_DESC = (
    "Look up a PAST order's status / tracking (order ids look like ORD-* or "
    "RCPT-*). Pass the user's order question as `query`."
)

SUBAGENTS: list[SubagentSpec] = [
    SubagentSpec(
        name="product_rec",
        description=_PRODUCT_REC_DESC,
        config=PRODUCT_REC_SPEC,
        snapshot=cart_quantities,
        build_input=with_cart_note,
        extract=extract_product_rec,
        summarize=summarize_product_rec,
        block="product_reco",
    ),
    SubagentSpec(
        name="checkout",
        description=_CHECKOUT_DESC,
        config=CHECKOUT_SPEC,
        build_input=checkout_input,
        extract=extract_checkout,
        summarize=summarize_checkout,
        block="checkout",
    ),
    SubagentSpec(
        name="order_status",
        description=_ORDER_STATUS_DESC,
        config=ORDER_STATUS_SPEC,
        extract=extract_order_status,
        block="order_status",
    ),
]

# sub-agent name -> writer block kind (consumed by build_blocks)
BLOCK_BY_SOP: dict[str, str | None] = {s.name: s.block for s in SUBAGENTS}

# delegate names, surfaced in the orchestrator_input trace frame
ROUTER_PROMPT_DELEGATES: list[str] = [s.name for s in SUBAGENTS]


# ===========================================================================
# builders (lazy singletons — importing this module needs no API key)
# ===========================================================================
_SUBAGENT_TOOLS: dict[str, Any] | None = None
_WRITER: Any | None = None


def subagent_tools() -> dict[str, Any]:
    """Build (once) the sub-agents wrapped as the orchestrator's delegate tools."""
    global _SUBAGENT_TOOLS
    if _SUBAGENT_TOOLS is None:
        _SUBAGENT_TOOLS = build_subagent_tools(SUBAGENTS, build_agent=build_agent)
    return _SUBAGENT_TOOLS


def build_orchestrator(cart_empty: bool) -> Any:
    """Compile the router orchestrator with the sub-agents wired in as delegates.

    While the cart is empty the checkout delegate is withheld (the empty-cart
    guard), so a first "add X" cannot route to checkout.
    """
    tools = subagent_tools()
    delegates = [
        fn for name, fn in tools.items() if not (cart_empty and name == "checkout")
    ]
    return build_agent(ORCHESTRATOR_SPEC, tools=delegates)


def build_writer() -> Any:
    """Compile the no-tools writer (its tokens stream to the client)."""
    global _WRITER
    if _WRITER is None:
        _WRITER = build_agent(WRITER_SPEC)
    return _WRITER


__all__ = [
    "PRODUCT_REC_SPEC",
    "CHECKOUT_SPEC",
    "ORDER_STATUS_SPEC",
    "WRITER_SPEC",
    "ORCHESTRATOR_SPEC",
    "SUBAGENTS",
    "BLOCK_BY_SOP",
    "ROUTER_PROMPT_DELEGATES",
    "subagent_tools",
    "build_orchestrator",
    "build_writer",
]
