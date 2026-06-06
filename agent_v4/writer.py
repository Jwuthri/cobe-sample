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
from typing import Literal

from agent_v4 import ids
from agent_v4.llm import writer_model_name
from agent_v4.state import AgentState
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
        * Don't talk about the cart unless ``cart.items`` is non-empty
          AND a step actually added something.

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


def writer(state: AgentState) -> Command:
    """Outer-graph writer node. Produces the single user-facing draft.

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
    return Command(goto="checkout_gate", update={"draft_response": text or "(no response)"})


__all__ = ["writer", "_pick_mode", "_build_writer_payload"]
