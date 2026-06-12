"""Build the writer's input payload from verified step results + the live cart.

Ported from agent_v4_1's ``build_writer_payload`` / ``pick_mode``; reads the
framework-agnostic ``Msg`` transcript instead of LangChain messages. The writer
composes from these grounded facts — never from raw tool output it could misread.
"""

from __future__ import annotations

import json
from typing import Literal

from openai_agent_v1.core.messages import Msg
from openai_agent_v1.core.step_result import StepResult
from openai_agent_v1.shopping.extractors import CHECKOUT

WriterMode = Literal["smalltalk", "checkout", "info"]

# Recent transcript turns the writer sees (bounded — its payload is volatile and
# never hits the prompt cache, so this keeps per-turn cost flat).
WRITER_HISTORY_MSGS = 8


def _format_history(messages: list[Msg]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m.role == "human":
            out.append({"role": "user", "content": str(m.content)})
        elif m.role == "ai" and isinstance(m.content, str) and m.content.strip():
            out.append({"role": "assistant", "content": m.content})
    return out


def _last_user_message(messages: list[Msg]) -> str:
    for m in reversed(messages):
        if m.role == "human":
            return str(m.content)
    return ""


def pick_mode(step_results: list[StepResult]) -> WriterMode:
    sops = {r.sop for r in step_results}
    if CHECKOUT in sops:
        return "checkout"
    if sops:
        return "info"
    return "smalltalk"


def _cart_summary_for_checkout(cart) -> dict:
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
    if not cart.items:
        return None
    return {
        "items": [
            {"id": i.product_id, "name": i.name, "qty": i.quantity, "price": str(i.unit_price)}
            for i in cart.items
        ],
        "subtotal": str(cart.subtotal),
    }


def build_writer_payload(
    messages: list[Msg], step_results: list[StepResult], cart
) -> tuple[str, WriterMode]:
    """Render the writer's input. Returns (payload_json, mode)."""
    mode = pick_mode(step_results)
    payload: dict = {
        "mode": mode,
        "user_message": _last_user_message(messages),
        "recent_conversation": _format_history(messages[-WRITER_HISTORY_MSGS:]),
        "step_results": [r.model_dump(mode="json", exclude={"recall"}) for r in step_results],
    }
    if mode == "checkout":
        payload["cart"] = _cart_summary_for_checkout(cart)
    elif mode == "info":
        cart_info = _cart_summary_for_info(cart)
        if cart_info is not None:
            payload["cart"] = cart_info
    return json.dumps(payload, ensure_ascii=False, indent=2), mode


__all__ = ["build_writer_payload", "pick_mode", "WriterMode", "WRITER_HISTORY_MSGS"]
