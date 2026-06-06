"""The writer — the single voice that talks to the user (Agno port).

Wrappers/steps produce structured ``StepResult`` records; the writer
composes ONE reply per turn in one of three modes (smalltalk / info /
checkout). v2 used a ``langchain_openai.ChatOpenAI`` call inside a graph
node; v3 uses an Agno ``Agent`` whose ``instructions`` are the writer
system prompt and whose ``input`` is the JSON payload.

The gate + validator that guard the draft live in :mod:`agent_v3.workflow`
(the ``compose`` step), mirroring v2 where they were graph nodes.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from agno.agent import Agent

from agent_v3.checkout import Cart
from agent_v3.models import writer_model
from agent_v3.sop_names import SOPName
from agent_v3.state import last_user_message

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
      The checkout SOP ran. Use ``cart`` and ``step_results``:
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


def _pick_mode(session_state: dict[str, Any]) -> WriterMode:
    """Decide which framing the writer should use."""
    sops_this_turn = {r.get("sop") for r in session_state.get("step_results", [])}
    if SOPName.CHECKOUT.value in sops_this_turn:
        return "checkout"
    if sops_this_turn:  # product_rec or order_status ran
        return "info"
    return "smalltalk"


def _cart_summary_for_checkout(cart: Cart) -> dict[str, Any]:
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


def _cart_summary_for_info(cart: Cart) -> dict[str, Any] | None:
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


def _build_writer_payload(session_state: dict[str, Any], cart: Cart) -> tuple[str, WriterMode]:
    """Render the writer's input. Returns (payload_json, mode)."""
    mode = _pick_mode(session_state)
    payload: dict[str, Any] = {
        "mode": mode,
        "user_message": last_user_message(session_state),
        "step_results": list(session_state.get("step_results", [])),
    }
    if mode == "checkout":
        payload["cart"] = _cart_summary_for_checkout(cart)
    elif mode == "info":
        info_cart = _cart_summary_for_info(cart)
        if info_cart is not None:
            payload["cart"] = info_cart
    # smalltalk: no cart payload at all.
    return json.dumps(payload, ensure_ascii=False, indent=2), mode


_WRITER: Agent | None = None


def build_writer_agent() -> Agent:
    return Agent(
        name="writer",
        model=writer_model(),
        instructions=WRITER_SYSTEM,
        telemetry=False,
    )


def _writer_agent() -> Agent:
    global _WRITER
    if _WRITER is None:
        _WRITER = build_writer_agent()
    return _WRITER


def generate_draft(session_state: dict[str, Any], cart: Cart, correction: str | None = None) -> str:
    """Compose the single user-facing draft for this turn."""
    payload, _mode = _build_writer_payload(session_state, cart)
    if correction:
        payload = f"{payload}\n\nIMPORTANT CORRECTION: {correction}"
    resp = _writer_agent().run(input=payload)
    content = resp.content
    text = content.strip() if isinstance(content, str) else str(content)
    return text or "(no response)"
