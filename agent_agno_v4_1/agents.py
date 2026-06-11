"""Assemble the Agno agents: 3 member sub-agents, the router Team, the writer.

Topology (mirrors agent_v4_1 on Agno primitives):

    Team(coordinate)  ── router/leader, delegates to ↓
      ├─ product_rec   (browse + cart edits)
      ├─ checkout      (drive the purchase; per-turn cart anchor)
      └─ order_status  (past-order lookup)
    writer (separate Agent, no tools) ── the streamed, user-facing voice

Members carry **dynamic instructions** (callables that read ``run_context``): the
checkout member injects the deterministic "Checkout progress" anchor every turn
(the cart is the source of truth), and product_rec appends a live cart note.
"""

from __future__ import annotations

from typing import Any

from agno.agent import Agent
from agno.team.mode import TeamMode
from agno.team.team import Team

from agent_agno_v4_1.context import ShoppingContext, ctx_from
from agent_agno_v4_1.extractors import checkout_anchor_text
from agent_agno_v4_1.models import resolve_model
from agent_agno_v4_1.tools import CHECKOUT_TOOLS, ORDER_STATUS_TOOLS, PRODUCT_REC_TOOLS
from agent_v4_1.shopping.prompts import (
    CHECKOUT_PROMPT,
    ORDER_STATUS_PROMPT,
    PRODUCT_REC_PROMPT,
    WRITER_SYSTEM,
)

# Routing rules — ported from agent_v4_1 ROUTER_PROMPT, minus the LangGraph-only
# "emit DONE" sentinel (a Team leader doesn't need it). The leader's own final
# text is discarded; the dedicated writer composes the user reply.
TEAM_LEADER_INSTRUCTIONS = [
    "You coordinate a shopping assistant with three specialist members:",
    "  - product_rec: search/catalog, serviceability ('do you ship to 94110?'), and "
    "ALL cart-content edits (add/remove/change quantity, 'what's in my cart'). Adding "
    "an item is the cue to move toward checkout next.",
    "  - checkout: drive an in-progress purchase — identity, address, delivery, payment, "
    "and place the order ONLY on an explicit 'yes'. Needs items already in the cart.",
    "  - order_status: look up a PAST order's status/tracking (ids look like ORD-* or RCPT-*).",
    "Routing rules (handle EVERY distinct request in the message — a compound message "
    "like 'find a green cap and check order ORD-7' needs TWO members):",
    "  1. Empty cart + any shopping intent ('add X', 'buy X', 'I want X') -> product_rec.",
    "  2. Cart non-empty and the user is giving checkout data (name, address, delivery "
    "option, payment method, or 'yes'/'confirm') -> checkout.",
    "  3. Cart edits or cart questions ('remove the hoodie', 'make it 2', 'what's in my "
    "cart') -> product_rec, even mid-checkout. Checkout cannot add or remove items.",
    "  4. Generic browse questions mid-checkout ('what else do you sell', 'do you ship to X') "
    "-> product_rec.",
    "  5. Past-order tracking -> order_status.",
    "  6. Smalltalk / greetings / off-topic / 'what can you do' -> delegate to NO member.",
    "Give each member a short, self-contained instruction. Never invent a product id, an "
    "order id, or a request the user didn't make. Keep your own final text to one short line "
    "— a separate writer produces the customer-facing reply.",
]


# =============================================================================
# member factories — dynamic instructions read the live cart via run_context
# =============================================================================
def _product_rec_instructions(run_context: Any) -> str:
    ctx = ctx_from(run_context)
    cart = ctx.cart_service.cart
    if not cart.items:
        return PRODUCT_REC_PROMPT
    note = (
        "\n\nCurrent cart: "
        + "; ".join(f"{i.product_id} {i.name} x{i.quantity}" for i in cart.items)
        + ". To edit it, use remove_item / set_quantity — do NOT search the catalog to "
        "remove or change an item already in the cart."
    )
    return PRODUCT_REC_PROMPT + note


def _checkout_instructions(run_context: Any) -> str:
    cart = ctx_from(run_context).cart_service.cart
    # The deterministic progress anchor is the heart of the checkout flow.
    return CHECKOUT_PROMPT + "\n\n" + checkout_anchor_text(cart)


def build_members() -> list[Agent]:
    product_rec = Agent(
        name="product_rec",
        role="Browse the catalog, answer serviceability, and edit the cart.",
        model=resolve_model(0.0),
        tools=PRODUCT_REC_TOOLS,
        instructions=_product_rec_instructions,
    )
    checkout = Agent(
        name="checkout",
        role="Drive an in-progress purchase from identity to payment to confirmation.",
        model=resolve_model(0.0),
        tools=CHECKOUT_TOOLS,
        instructions=_checkout_instructions,
    )
    order_status = Agent(
        name="order_status",
        role="Look up a past order's status / tracking.",
        model=resolve_model(0.0),
        tools=ORDER_STATUS_TOOLS,
        instructions=ORDER_STATUS_PROMPT,
    )
    return [product_rec, checkout, order_status]


def build_team(members: list[Agent] | None = None) -> Team:
    return Team(
        name="shopping",
        mode=TeamMode.coordinate,
        model=resolve_model(0.0),
        members=members or build_members(),
        instructions=TEAM_LEADER_INSTRUCTIONS,
        show_members_responses=True,
        store_member_responses=True,
    )


def build_writer() -> Agent:
    """The no-tools, user-facing voice. Its tokens stream straight to the client."""
    return Agent(
        name="writer",
        model=resolve_model(0.3),
        instructions=WRITER_SYSTEM,
        markdown=False,
    )


__all__ = [
    "TEAM_LEADER_INSTRUCTIONS",
    "build_members",
    "build_team",
    "build_writer",
    "ShoppingContext",
]
