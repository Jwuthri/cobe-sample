"""``TurnContext`` — the per-turn state shared by the orchestrator and its tools.

One instance is created per user turn and passed to
``orchestrator.astream(..., context=ctx)``. LangChain forwards the same object to
each tool's ``ToolRuntime``, and the sub-agent tool forwards it again into the
sub-agent's ``.stream(context=ctx)`` — so one shared state (the cart, in the
shopping tenant) is visible end-to-end within a turn with no copy-back.

``TurnContext`` is the generic base (ids + a ``step_results`` accumulator + a usage
tally). A tenant subclasses it to carry domain handles (see
:class:`lg_agent.shopping.context.ShoppingContext`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage

from lg_agent.core.step import StepResult


def zero_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "llm_calls": 0}


def add_message_usage(sink: dict[str, int], messages: list[Any]) -> None:
    """Accumulate token usage + LLM-call count from a run's AIMessages.

    Sub-agent runs happen inside the orchestrator's tools and their messages are
    otherwise discarded, so we tally their ``usage_metadata`` into the shared
    context to get a true per-turn cost.
    """
    for m in messages or []:
        if not isinstance(m, AIMessage):
            continue
        um = getattr(m, "usage_metadata", None) or {}
        if um:
            sink["input_tokens"] += int(um.get("input_tokens", 0) or 0)
            sink["output_tokens"] += int(um.get("output_tokens", 0) or 0)
            sink["cached_tokens"] += int(
                (um.get("input_token_details") or {}).get("cache_read", 0) or 0
            )
            sink["llm_calls"] += 1


@dataclass
class TurnContext:
    """Static-per-turn config + a mutable result accumulator."""

    user_id: str = "anonymous"
    session_id: str = "session"
    step_results: list[StepResult] = field(default_factory=list)
    skills_loaded: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=zero_usage)
    # When True, sub-agent tools emit deep-trace events on the custom stream.
    debug: bool = False

    def debug_view(self) -> dict[str, Any]:
        """A JSON-safe snapshot of the mutable runtime state (for the trace UI).

        Tenants override to add domain handles (see
        :meth:`lg_agent.shopping.context.ShoppingContext.debug_view`).
        """
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "skills_loaded": list(self.skills_loaded),
            "usage": dict(self.usage),
            "step_results": [r.model_dump(mode="json") for r in self.step_results],
        }

    def routing_context(self) -> dict[str, str]:
        """Live structured state the orchestrator should see to resolve references.

        Returns ``{label: llm_ready_text}``. The base carries no domain state — a
        tenant overrides this to surface, e.g., the current cart. The orchestrator
        merges these *live* blocks with the *persisted* per-step ``recall`` snippets
        to build its routing memo — all domain-agnostic.
        """
        return {}


__all__ = ["TurnContext", "add_message_usage", "zero_usage"]
