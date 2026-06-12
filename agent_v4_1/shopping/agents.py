"""The shopping agents — declared as plain dicts in the v4.1 config schema.

Five definitions:
  * three sub-agents (product_rec / checkout / order_status), each a config dict
    plus a :class:`SubagentSpec` wiring its domain hooks;
  * the writer (a no-tools agent — the user-facing voice);
  * the orchestrator (routes to the sub-agent tools, then emits DONE).

The dicts ARE the public surface: an agent's full definition is data. Note the
checkout config declares the ``cart_anchor`` middleware (the progress block) and
NO skills — the cart drives its flow, so v4's load_skill chain is gone.
"""

from __future__ import annotations

from agent_v4_1.core.subagent import SubagentSpec
from agent_v4_1.shopping.extractors import (
    cart_quantities,
    checkout_input,
    extract_checkout,
    extract_order_status,
    extract_product_rec,
    summarize_checkout,
    summarize_product_rec,
    with_cart_note,
)
from agent_v4_1.shopping.prompts import (
    CHECKOUT_PROMPT,
    ORDER_STATUS_PROMPT,
    PRODUCT_REC_PROMPT,
    ROUTER_PROMPT,
    WRITER_SYSTEM,
)
from agent_v4_1.shopping.tools import CHECKOUT_TOOLS, ORDER_STATUS_TOOLS, PRODUCT_REC_TOOLS


def _registry_tools(tools) -> list[dict]:
    return [{"kind": "registry", "name": t.name} for t in tools]


# =============================================================================
# MODEL — the one knob. Every agent below declares it explicitly in its `model`
# block (no hidden env fallback). Format is "provider:model"; override per agent
# by writing a literal in that agent's block instead of MODEL.
# =============================================================================
MODEL = "openai:gpt-5.4-mini"


# =============================================================================
# sub-agent configs
# =============================================================================
PRODUCT_REC_AGENT = {
    "name": "product_rec",
    "description": "Browse + cart management: search, lookup, serviceability, add/remove/qty/view.",
    "system_prompt": PRODUCT_REC_PROMPT,
    "model": {"provider_model": MODEL, "temperature": 0.0},
    "tools": _registry_tools(PRODUCT_REC_TOOLS),
    "middleware": [{"name": "log_tool_calls", "params": {"log_prefix": "product_rec"}}],
}

CHECKOUT_AGENT = {
    "name": "checkout",
    "description": "Drive an in-progress purchase from identity to payment to confirmation.",
    "system_prompt": CHECKOUT_PROMPT,
    "model": {"provider_model": MODEL, "temperature": 0.0},
    "tools": _registry_tools(CHECKOUT_TOOLS),
    "middleware": [
        {"name": "cart_anchor", "params": {}},
        {"name": "log_tool_calls", "params": {"log_prefix": "checkout"}},
    ],
}

ORDER_STATUS_AGENT = {
    "name": "order_status",
    "description": "Look up a past order's status / tracking.",
    "system_prompt": ORDER_STATUS_PROMPT,
    "model": {"provider_model": MODEL, "temperature": 0.0},
    "tools": _registry_tools(ORDER_STATUS_TOOLS),
    "middleware": [{"name": "log_tool_calls", "params": {"log_prefix": "order_status"}}],
}


# =============================================================================
# writer (no tools) + orchestrator (routes to sub-agent tools)
# =============================================================================
WRITER_AGENT = {
    "name": "writer",
    "description": "Compose the single user-facing reply from verified step results + cart.",
    "system_prompt": WRITER_SYSTEM,
    "model": {"provider_model": MODEL, "temperature": 0.3},
    "tools": [],
}

ORCHESTRATOR_AGENT = {
    "name": "orchestrator",
    "description": "Route the user's message to its sub-agents, then emit DONE.",
    "system_prompt": ROUTER_PROMPT,
    "model": {"provider_model": MODEL, "temperature": 0.0},
    # NB: no `tools` here. The orchestrator's delegates are the sub-agents in
    # SUBAGENTS, wired in by build_orchestrator() via build_agent(delegates=...).
    # Sub-agents are not registry tools — they carry Python extractors and live
    # in SUBAGENTS, not in this config or the global TOOLS registry.
    "middleware": [
        {"name": "empty_cart_guard", "params": {}},
        {"name": "tool_call_limit", "params": {"run_limit": 4, "exit_behavior": "end"}},
        # log_tool_calls makes each sub-agent call visible as a router/agent event
        # (classify_custom maps a tool_start/tool_end on a sub-agent name to those).
        {"name": "log_tool_calls", "params": {"log_prefix": "orchestrator"}},
    ],
}


# =============================================================================
# sub-agent specs (config + domain hooks) — what the platform wraps as tools
# =============================================================================
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
        config=PRODUCT_REC_AGENT,
        snapshot=cart_quantities,
        build_input=with_cart_note,
        extract=extract_product_rec,
        summarize=summarize_product_rec,
        block="product_reco",
    ),
    SubagentSpec(
        name="checkout",
        description=_CHECKOUT_DESC,
        config=CHECKOUT_AGENT,
        build_input=checkout_input,
        extract=extract_checkout,
        summarize=summarize_checkout,
        block="checkout",
    ),
    SubagentSpec(
        name="order_status",
        description=_ORDER_STATUS_DESC,
        config=ORDER_STATUS_AGENT,
        extract=extract_order_status,
        block="order_status",
    ),
]

# sub-agent name -> writer block kind (consumed by build_blocks)
BLOCK_BY_SOP: dict[str, str | None] = {s.name: s.block for s in SUBAGENTS}

__all__ = [
    "PRODUCT_REC_AGENT",
    "CHECKOUT_AGENT",
    "ORDER_STATUS_AGENT",
    "WRITER_AGENT",
    "ORCHESTRATOR_AGENT",
    "SUBAGENTS",
    "BLOCK_BY_SOP",
]
