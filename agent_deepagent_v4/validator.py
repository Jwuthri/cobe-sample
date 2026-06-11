"""Validator middleware — the structural safety net on the orchestrator.

Mirrors agent_v4's ``validator`` node: a minimal, content-agnostic guarantee
that the turn ends with a non-empty customer-facing message. If the orchestrator
somehow produced no text (e.g. it ended on a tool call), the validator
substitutes a graceful fallback so the customer never sees an empty turn.

Confirmation-safety (never claiming the order is placed unless the cart says so)
is enforced upstream by three things working together — the ``blockers()`` gate,
the ``confirm_checkout`` human-approval interrupt, and the writer's prompt — so
the validator deliberately stays out of content judgement (no regexes), exactly
as agent_v4 settled on after its regex "gate" produced false positives.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from agent_deepagent_v4.messages import text_of

_FALLBACK = "Sorry, I couldn't put together a response just now. Could you rephrase that?"


class ResponseValidatorMiddleware(AgentMiddleware):
    """Guarantee a non-empty final assistant message."""

    def after_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:  # noqa: ANN401
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        is_ai = isinstance(last, AIMessage) or getattr(last, "type", None) == "ai"
        if is_ai and text_of(last).strip():
            return None
        # No usable final text → append a graceful fallback (add_messages appends).
        return {"messages": [AIMessage(content=_FALLBACK)]}
