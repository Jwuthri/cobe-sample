"""Build the writer's grounded input payload.

The cart-summary shaping (``pick_mode``, ``_cart_summary_for_checkout``,
``_cart_summary_for_info``) is pure and reused verbatim from ``agent_v4_1``; only
the transcript format differs (plain ``{role, content}`` dicts instead of
LangChain message objects), so this package has no LangChain dependency.
"""

from __future__ import annotations

import json

from agent_v4_1.core.step_result import StepResult
from agent_v4_1.shopping.writer_payload import (
    WRITER_HISTORY_MSGS,
    WriterMode,
    _cart_summary_for_checkout,
    _cart_summary_for_info,
    pick_mode,
)


def _format_history(history: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in history:
        role, content = m.get("role"), m.get("content")
        if role == "user" and content:
            out.append({"role": "user", "content": str(content)})
        elif role == "assistant" and isinstance(content, str) and content.strip():
            out.append({"role": "assistant", "content": content})
    return out


def _last_user_message(history: list[dict]) -> str:
    for m in reversed(history):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def build_writer_payload(
    history: list[dict], step_results: list[StepResult], cart
) -> tuple[str, WriterMode]:
    """Render the writer's input from grounded facts. Returns (payload_json, mode)."""
    mode = pick_mode(step_results)
    payload: dict = {
        "mode": mode,
        "user_message": _last_user_message(history),
        "recent_conversation": _format_history(history[-WRITER_HISTORY_MSGS:]),
        "step_results": [r.model_dump(mode="json") for r in step_results],
    }
    if mode == "checkout":
        payload["cart"] = _cart_summary_for_checkout(cart)
    elif mode == "info":
        cart_info = _cart_summary_for_info(cart)
        if cart_info is not None:
            payload["cart"] = cart_info
    return json.dumps(payload, ensure_ascii=False, indent=2), mode


__all__ = ["build_writer_payload"]
