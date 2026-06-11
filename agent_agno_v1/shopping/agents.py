"""The shopping agents — declared as plain dicts in the config schema.

Five definitions:
  * three members (product_rec / checkout / order_status), each a config dict
    naming its registry tools + a stable ``id`` (the leader's delegate target);
  * the supervisor TEAM config (the speaking coordinator — routes AND writes).

The dicts ARE the public surface: an agent's full definition is data. The
members carry NO ``add_item`` on checkout (a double-add is structurally
impossible), and there are no skills (the cart drives the checkout flow).
"""

from __future__ import annotations

from agent_agno_v1.shopping.extractors import CHECKOUT, ORDER_STATUS, PRODUCT_REC
from agent_agno_v1.shopping.prompts import (
    CHECKOUT_PROMPT,
    ORDER_STATUS_PROMPT,
    PRODUCT_REC_PROMPT,
    SUPERVISOR_PROMPT,
)
from agent_agno_v1.shopping.tools import (
    CHECKOUT_TOOLS,
    ORDER_STATUS_TOOLS,
    PRODUCT_REC_TOOLS,
)


def _registry_tools(tools) -> list[dict]:
    return [{"kind": "registry", "name": t.name} for t in tools]


# =============================================================================
# member configs
# =============================================================================
PRODUCT_REC_AGENT = {
    "name": "Product Rec",
    "id": PRODUCT_REC,
    "role": "Browse + cart management: search, lookup, serviceability, add/remove/qty/view.",
    "system_prompt": PRODUCT_REC_PROMPT,
    "model": {"temperature": 0.0},
    "tools": _registry_tools(PRODUCT_REC_TOOLS),
    "tool_call_limit": 6,
}

CHECKOUT_AGENT = {
    "name": "Checkout",
    "id": CHECKOUT,
    "role": "Drive an in-progress purchase from identity to payment to confirmation.",
    "system_prompt": CHECKOUT_PROMPT,
    "model": {"temperature": 0.0},
    "tools": _registry_tools(CHECKOUT_TOOLS),
    "tool_call_limit": 12,
}

ORDER_STATUS_AGENT = {
    "name": "Order Status",
    "id": ORDER_STATUS,
    "role": "Look up a past order's status / tracking.",
    "system_prompt": ORDER_STATUS_PROMPT,
    "model": {"temperature": 0.0},
    "tools": _registry_tools(ORDER_STATUS_TOOLS),
    "tool_call_limit": 4,
}


# =============================================================================
# supervisor team (the speaking coordinator)
# =============================================================================
SUPERVISOR_TEAM = {
    "name": "Shopping Supervisor",
    "id": "supervisor",
    "description": "Routes the user's message to its members, then composes the single reply.",
    "system_prompt": SUPERVISOR_PROMPT,
    # Slightly warm: the leader authors the user-facing prose (v4.1's writer voice).
    "model": {"temperature": 0.3},
    # Cap delegations per turn (v4.1's orchestrator run_limit analogue).
    "tool_call_limit": 6,
}

# ordered member configs (the order the leader sees them in)
MEMBER_CONFIGS = [PRODUCT_REC_AGENT, CHECKOUT_AGENT, ORDER_STATUS_AGENT]

# member id -> writer block kind (consumed by build_blocks)
BLOCK_BY_SOP: dict[str, str | None] = {
    PRODUCT_REC: "product_reco",
    CHECKOUT: "checkout",
    ORDER_STATUS: "order_status",
}

__all__ = [
    "PRODUCT_REC_AGENT",
    "CHECKOUT_AGENT",
    "ORDER_STATUS_AGENT",
    "SUPERVISOR_TEAM",
    "MEMBER_CONFIGS",
    "BLOCK_BY_SOP",
]
