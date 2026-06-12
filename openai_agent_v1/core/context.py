"""Per-turn runtime context shared by the orchestrator and every sub-agent tool.

One instance is created per user turn and passed to ``Runner.run(..., context=ctx)``.
The OpenAI Agents SDK wraps it in a ``RunContextWrapper`` and forwards the SAME
object to each tool; the sub-agent tool forwards it again into its own
``Runner.run(..., context=ctx)`` — so a single shared state (the cart, in the
shopping tenant) is visible end-to-end within a turn with no copy-back.

``TurnContext`` is the generic base (ids + a ``step_results`` accumulator + a
usage tally + a live **event bus**); a tenant subclasses it to carry domain
handles (see :class:`openai_agent_v1.shopping.context.ShoppingContext`).

The **event bus** replaces langgraph's custom stream: the SDK has no built-in
channel for a tool to push UI events to the turn loop, so sub-agent tools call
``ctx.emit(event)`` and the session drains the queue live.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


def zero_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "llm_calls": 0}


def add_run_usage(sink: dict[str, int], run_result: Any) -> None:
    """Accumulate token usage from a sub-agent ``RunResult`` into the shared tally.

    Sub-agent runs happen inside the orchestrator's tools; their usage would
    otherwise be lost, so we fold each run's ``context_wrapper.usage`` into the
    turn context to get a true per-turn cost.
    """
    wrapper = getattr(run_result, "context_wrapper", None)
    usage = getattr(wrapper, "usage", None)
    if usage is None:
        return
    sink["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
    sink["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
    details = getattr(usage, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    sink["cached_tokens"] += int(cached or 0)
    sink["llm_calls"] += int(getattr(usage, "requests", 0) or 0)


@dataclass
class TurnContext:
    """Static-per-turn config + a mutable result accumulator + a live event bus."""

    user_id: str = "anonymous"
    session_id: str = "session"
    step_results: list[Any] = field(default_factory=list)
    skills_loaded: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=zero_usage)
    # When True, sub-agent tools emit deep-trace events on the bus.
    debug: bool = False
    # Live event channel: the session installs an asyncio.Queue here per turn and
    # sub-agent tools push UI events to it via ``emit``. ``None`` outside a turn
    # (e.g. a plain ``Runner.run`` in a test) → ``emit`` is a no-op.
    bus: asyncio.Queue | None = None
    # Long-term memory store handle (the shopping tools read it for order history).
    store: Any = None

    def emit(self, event: dict) -> None:
        """Push a UI event onto the live bus (no-op when no bus is installed)."""
        if self.bus is not None:
            self.bus.put_nowait(event)

    def debug_view(self) -> dict[str, Any]:
        """A JSON-safe snapshot of the mutable runtime state (for the trace UI).

        Tenants override to add domain handles (see
        :meth:`openai_agent_v1.shopping.context.ShoppingContext.debug_view`).
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
        tenant overrides this to surface, e.g., the current cart. The session
        merges these *live* blocks with the *persisted* per-step ``recall``
        snippets to build the orchestrator's routing memo — all domain-agnostic.
        """
        return {}


__all__ = ["TurnContext", "add_run_usage", "zero_usage"]
