"""The writer — the single voice that talks to the user.

Wrappers no longer produce user-facing text; they produce structured
``StepResult`` records. The writer composes ONE reply per turn,
choosing one of three **modes**:

  - ``smalltalk``  — no leaf ran this turn (greeting, off-topic, etc.).
                     Reply briefly and conversationally. Do NOT mention
                     the cart, blockers, or checkout in any way.
  - ``checkout``   — checkout was the active leaf. Surface cart state,
                     outstanding asks, totals, blockers as needed.
  - ``info``       — product_rec or order_status ran. Summarize their
                     step results. Don't mention the cart unless an
                     item was actually added.

This mode-aware framing is what prevents the writer from talking
about "your cart is empty" on a greeting turn.

The writer is a single LLM call (not a ``create_agent`` leaf), so its
``ModelConfig`` lives here rather than in :mod:`agent_v4.leaves`.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from agent_v4 import ids
from agent_v4.leaves import LEAVES_BY_NAME
from agent_v4.llm import writer_model_name
from agent_v4.output_schemas import (
    CheckoutBlock,
    OrderLine,
    OrderStatusBlock,
    ProductCard,
    ProductRecoBlock,
)
from agent_v4.state import AgentState
from agent_v4.tools.orders_db import get_order
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

WriterMode = Literal["smalltalk", "checkout", "info"]


WRITER_SYSTEM = """\
You are the customer-facing assistant in a multi-agent shopping
system. Other agents may have done work this turn; your job is to
compose ONE clear, concise message back to the user.

The input payload tells you which **mode** to use. Honor it strictly:

  - mode = "smalltalk"
      The user said something conversational, off-topic, or just hi.
      Reply briefly and warmly. DO NOT mention the cart, items,
      checkout, addresses, payment, or anything shop-related unless
      the user explicitly asked. If the user asked what you can do,
      mention you can help find products, place orders, and check
      order status — but in one short line.

  - mode = "info"
      product_rec or order_status ran. Use ``step_results[*].details``
      as the source of truth for what to show:
        * If ``details.serviceability`` is set, lead with that — quote
          the raw answer (or paraphrase it cleanly). Examples:
            - "Yes, we ship to 94110 (San Francisco, US). Options: 2h, 4h, next_day, standard."
            - "We don't currently ship to zip 99999."
        * If ``details.products`` is set, list THOSE products as a
          short bullet list with id, name, and price. Then ask which
          one they want (reply with a product id). NEVER invent
          products or use placeholder ids; if details.products is
          empty, say "I couldn't find anything matching that — try
          another search?".
        * If ``details.order`` is set, present that order info clearly.
        * If ``details.cart_edit`` is set, the user edited or viewed their
          cart. Confirm the change briefly (e.g. "Removed the Black
          Hoodie") and show the resulting cart from
          ``details.cart_edit.items`` (id, name, qty) plus the subtotal if
          present. If that items list is empty, say the cart is now empty.
        * Otherwise don't volunteer cart contents unless a step actually
          added or edited the cart (``details.added`` / ``details.cart_edit``).

  - mode = "checkout"
      The checkout leaf ran. Use ``cart`` and ``step_results``:
        * Summarize what happened (added item, captured address, etc.).
        * If ``step_results[*].asks`` is non-empty, list them clearly
          so the user knows exactly what to provide next.
        * If ``cart.grand_total`` is set, quote it as USD.
        * If ``cart.blockers`` has items, mention the ones the user
          can act on (the payload already pre-filtered to actionable
          ones, so just enumerate them).
        * **If ``cart.ready_to_confirm`` is true AND ``cart.confirmed``
          is false** (i.e. the cart is fully prepared but not yet
          placed): present a short order summary (items + total)
          and END the message with a clear yes/no confirmation
          prompt, e.g. "Reply 'yes' to place the order." This is the
          place where the user gives final approval — be explicit.
        * If ``cart.confirmed`` is true and ``cart.receipt_id`` is
          set, congratulate the user and quote the receipt id.

Universal rules:
  - Never invent facts. If a field is null/missing, don't reference it.
  - Friendly but brief. No emoji unless the user used one.
  - Don't ask for things the user already provided this conversation.
  - When listing products or orders, copy the ids EXACTLY as given.
  - Structured cards (product lists, order details, the order summary) are
    rendered to the user separately from your text. Introduce them naturally
    in prose (e.g. "Here are the hoodies:") but do NOT re-dump every id, price,
    or field in the message — the cards already show them.
"""


def _pick_mode(state: AgentState) -> WriterMode:
    """Decide which framing the writer should use."""
    sops_this_turn = {r.sop for r in state.step_results}
    if ids.CHECKOUT in sops_this_turn:
        return "checkout"
    if sops_this_turn:  # product_rec or order_status ran
        return "info"
    return "smalltalk"


def _cart_summary_for_checkout(cart) -> dict:
    """Cart fields the writer needs when mode='checkout'."""
    USER_ACTIONABLE = {
        "empty_cart",
        "missing_identity",
        "missing_address",
        "not_serviceable",
        "missing_delivery_option",
        "unserviceable_delivery_option",
        "missing_payment",
        "missing_card_token",
        "invalid_promo",
    }
    blockers = [
        {"code": b.code, "message": b.message} for b in cart.blockers() if b.code in USER_ACTIONABLE
    ]
    return {
        "step": cart.step.value,
        "items": [
            {"id": i.product_id, "name": i.name, "qty": i.quantity, "price": str(i.unit_price)}
            for i in cart.items
        ],
        "subtotal": str(cart.subtotal),
        "address": cart.address.model_dump(),
        "customer": cart.customer.model_dump(),
        "delivery_option": cart.delivery_option,
        "serviceable": cart.serviceable,
        "serviceable_options": list(cart.serviceable_options),
        "shipping": (
            {"cost": str(cart.shipping.cost), "eta_hours": cart.shipping.eta_hours}
            if cart.shipping_is_fresh()
            else None
        ),
        "tax": {"amount": str(cart.tax.amount)} if cart.tax_is_fresh() else None,
        "payment_method": cart.payment_method,
        "card_token_set": bool(cart.card_token),
        "promo": (
            {"code": cart.promo.code, "discount": str(cart.promo.discount)} if cart.promo else None
        ),
        "grand_total": str(cart.grand_total) if cart.grand_total is not None else None,
        "blockers": blockers,
        "ready_to_confirm": cart.ready_to_confirm(),
        "confirmed": cart.confirmed,
        "receipt_id": cart.receipt_id,
    }


def _cart_summary_for_info(cart) -> dict | None:
    """Minimal cart info for info mode — only if items exist."""
    if not cart.items:
        return None
    return {
        "items": [
            {"id": i.product_id, "name": i.name, "qty": i.quantity, "price": str(i.unit_price)}
            for i in cart.items
        ],
        "subtotal": str(cart.subtotal),
    }


def _build_writer_payload(state: AgentState) -> tuple[str, WriterMode]:
    """Render the writer's input. Returns (payload_json, mode)."""
    mode = _pick_mode(state)
    payload: dict = {
        "mode": mode,
        "user_message": state.last_user_message(),
        "step_results": [r.model_dump(mode="json") for r in state.step_results],
    }
    if mode == "checkout":
        payload["cart"] = _cart_summary_for_checkout(state.cart)
    elif mode == "info":
        cart = _cart_summary_for_info(state.cart)
        if cart is not None:
            payload["cart"] = cart
    # smalltalk: no cart payload at all.

    return json.dumps(payload, ensure_ascii=False, indent=2), mode


# =============================================================================
# Rich-reply blocks — typed payloads assembled DETERMINISTICALLY from what the
# leaves already produced (step_results[*].details + the cart). The LLM only
# writes the prose `message`; ids/prices in blocks are verbatim (no hallucination).
# =============================================================================
def _order_from_raw(raw: str | None):
    """Resolve a structured Order from an order_status tool's raw text, if any."""
    if not raw:
        return None
    match = re.search(r"(ORD-\d+)", raw, re.IGNORECASE)
    if not match:
        return None
    return get_order(match.group(1))


def _checkout_block(cart, asks: list[str]) -> CheckoutBlock:
    items = [
        OrderLine(
            id=i.product_id,
            name=i.name,
            qty=i.quantity,
            line_total=f"{i.line_total:.2f}",
        )
        for i in cart.items
    ]
    return CheckoutBlock(
        items=items,
        subtotal=f"{cart.subtotal:.2f}",
        grand_total=f"{cart.grand_total:.2f}" if cart.grand_total is not None else None,
        ready_to_confirm=cart.ready_to_confirm(),
        confirmed=cart.confirmed,
        receipt_id=cart.receipt_id,
        asks=list(asks),
    )


def build_blocks(state: AgentState) -> list[dict]:
    """Assemble the turn's typed blocks from step results + cart.

    One block per structured thing a leaf produced, in order. A turn with both
    a product_rec and an order_status step yields two blocks. Conversational /
    smalltalk turns (no qualifying details) yield ``[]``.
    """
    blocks: list = []
    checkout_done = False
    for sr in state.step_results:
        spec = LEAVES_BY_NAME.get(sr.sop)
        kind = spec.output_block if spec else None
        details = sr.details or {}

        if kind == "product_reco":
            products = [ProductCard(**p) for p in (details.get("products") or [])]
            added = list(details.get("added") or [])
            serv = details.get("serviceability")
            serv_raw = (
                serv.get("raw") if isinstance(serv, dict) else serv if isinstance(serv, str) else None
            )
            if products or added or serv_raw:
                blocks.append(
                    ProductRecoBlock(products=products, added_ids=added, serviceability=serv_raw)
                )
        elif kind == "order_status":
            raw = details.get("raw")
            if raw:
                blocks.append(OrderStatusBlock(order=_order_from_raw(raw), raw=raw))
        elif kind == "checkout" and not checkout_done:
            blocks.append(_checkout_block(state.cart, sr.asks))
            checkout_done = True

    return [b.model_dump(mode="json") for b in blocks]


def writer(state: AgentState) -> Command:
    """Outer-graph writer node. Produces the single user-facing draft.

    The model writes the prose ``message`` (kept in ``draft_response`` so every
    text path — gate, validator, emit, UI — is unchanged). Typed ``draft_blocks``
    are assembled deterministically alongside it.

    Confirmation handling: when the cart is ``ready_to_confirm`` but
    ``not confirmed``, the system prompt instructs the model to end
    its message with a clear "Reply 'yes' to place the order" line.
    """
    payload, _mode = _build_writer_payload(state)
    chat = ChatOpenAI(model=writer_model_name(), temperature=0.3)
    resp = chat.invoke(
        [
            SystemMessage(content=WRITER_SYSTEM),
            HumanMessage(content=payload),
        ]
    )
    text = (resp.content or "").strip() if isinstance(resp.content, str) else str(resp.content)
    return Command(
        goto="checkout_gate",
        update={
            "draft_response": text or "(no response)",
            "draft_blocks": build_blocks(state),
        },
    )


__all__ = ["writer", "build_blocks", "_pick_mode", "_build_writer_payload"]
