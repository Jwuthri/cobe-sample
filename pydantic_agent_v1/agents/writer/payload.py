"""The writer's input — a grounded JSON payload built from step results + cart.

The writer composes from these verified facts, never from raw tool output it could
misread. ``pick_mode`` chooses how it should present (smalltalk / info / checkout)
based on which workers ran this turn.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic_agent_v1.agents.names import CHECKOUT
from pydantic_agent_v1.runtime import StepResult

WriterMode = Literal["smalltalk", "checkout", "info"]

# Recent transcript turns the writer sees (bounded — its payload is volatile and
# never hits the prompt cache, so this keeps per-turn cost flat).
WRITER_HISTORY_MSGS = 8


def pick_mode(step_results: list[StepResult]) -> WriterMode:
    sops = {r.sop for r in step_results}
    if CHECKOUT in sops:
        return "checkout"
    if sops:
        return "info"
    return "smalltalk"


def _last_user_message(transcript: list[dict]) -> str:
    for m in reversed(transcript):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def _format_history(transcript: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in transcript:
        role, content = m.get("role"), str(m.get("content", ""))
        if role == "user":
            out.append({"role": "user", "content": content})
        elif role == "assistant" and content.strip():
            out.append({"role": "assistant", "content": content})
    return out


def _cart_summary_for_checkout(cart) -> dict:
    user_actionable = {
        "empty_cart", "missing_identity", "missing_address", "not_serviceable",
        "missing_delivery_option", "unserviceable_delivery_option", "missing_payment",
        "missing_card_token", "invalid_promo",
    }
    blockers = [
        {"code": b.code, "message": b.message} for b in cart.blockers() if b.code in user_actionable
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
    transcript: list[dict], step_results: list[StepResult], cart
) -> tuple[str, WriterMode]:
    """Render the writer's input. Returns ``(payload_json, mode)``."""
    mode = pick_mode(step_results)
    payload: dict = {
        "mode": mode,
        "user_message": _last_user_message(transcript),
        "recent_conversation": _format_history(transcript[-WRITER_HISTORY_MSGS:]),
        # ``recall`` is the orchestrator's private memory, not for the writer.
        "step_results": [r.model_dump(mode="json", exclude={"recall"}) for r in step_results],
    }
    if mode == "checkout":
        payload["cart"] = _cart_summary_for_checkout(cart)
    elif mode == "info":
        cart_info = _cart_summary_for_info(cart)
        if cart_info is not None:
            payload["cart"] = cart_info
    return json.dumps(payload, ensure_ascii=False, indent=2), mode
