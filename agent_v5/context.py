"""Per-turn runtime context shared by the supervisor and every subagent tool.

This is the v5 analogue of :class:`agent_v4.runtime.RuntimeContext`. The v4
version carried ``user_id`` / ``session_id`` / ``cart_service``; v5 adds a
``step_results`` accumulator because in the agent-as-tool topology there is no
outer LangGraph state to thread leaf results through — the subagent tools append
their typed :class:`~agent_v4.step_result.StepResult` here instead, and the
block-builder / writer read it back after the supervisor's tool loop finishes.

One instance is created per user turn and passed to
``supervisor.invoke(..., context=ctx)``. Because LangChain v1 forwards the same
``context`` object to a tool's ``ToolRuntime`` AND we forward it again into each
subagent's ``.invoke(context=ctx)``, the **same** ``cart_service`` (and therefore
the same live cart) is visible end-to-end within a turn — exactly the property the
v4 checkout tools already rely on (``runtime.context.cart_service``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_v4.checkout.service import CartService
from agent_v4.step_result import StepResult
from langchain_core.messages import AIMessage


def _zero_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "llm_calls": 0}


def add_message_usage(sink: dict[str, int], messages: list[Any]) -> None:
    """Accumulate token usage + LLM-call count from a run's AIMessages.

    Subagent invocations happen INSIDE the supervisor's tools and their messages
    are otherwise discarded, so we tally their ``usage_metadata`` here (into the
    shared context) to get a true per-turn cost — not just the supervisor's own
    calls. ``cached_tokens`` (OpenAI prompt-cache reads) is tracked so we can see
    how much of the input was served from cache vs reprocessed.
    """
    for m in messages or []:
        if not isinstance(m, AIMessage):
            continue
        um = getattr(m, "usage_metadata", None) or {}
        if um:
            sink["input_tokens"] += int(um.get("input_tokens", 0) or 0)
            sink["output_tokens"] += int(um.get("output_tokens", 0) or 0)
            sink["cached_tokens"] += int((um.get("input_token_details") or {}).get("cache_read", 0) or 0)
            sink["llm_calls"] += 1


@dataclass
class SupervisorContext:
    """Static-per-turn config + a mutable result accumulator.

    Fields:
      user_id / session_id: same roles as v4 (memory key + checkpointer thread).
      cart_service: the live cart handle for THIS turn; subagent tools mutate
        ``cart_service.cart`` directly and the changes are visible to the
        supervisor and the block-builder without any copy-back.
      step_results: appended to by each subagent tool. The deterministic
        block-builder (and, in the writer variant, the writer) read this list.
      skills_loaded: carried across turns so the checkout subagent doesn't
        re-load skills; mirrors ``AgentState.skills_loaded`` in v4.

    The field *names* are a superset of v4's ``RuntimeContext``, so the existing
    checkout tools (which read ``runtime.context.cart_service`` /
    ``runtime.context.user_id``) work against this class unchanged.
    """

    user_id: str
    session_id: str
    cart_service: CartService
    step_results: list[StepResult] = field(default_factory=list)
    skills_loaded: list[str] = field(default_factory=list)
    # Token/call tally for everything that runs INSIDE the supervisor's tools
    # (i.e. the subagents). The supervisor's own + the writer's usage are added
    # by the orchestrator in agent.py.
    subagent_usage: dict[str, int] = field(default_factory=_zero_usage)


__all__ = ["SupervisorContext", "add_message_usage"]
