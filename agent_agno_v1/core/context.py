"""Per-turn record the streaming session accumulates from Agno events.

Unlike agent_v4_1 (where a ``ToolRuntime`` context was threaded *into* every tool),
the live state in this Agno port travels through ``dependencies`` (the cart
service + memory store, by reference). What ``TurnContext`` collects instead is
the *observation* of a turn — the tool executions that streamed past and the
``StepResult`` s distilled from them — so the session can build deterministic
blocks and emit ``step`` events. It is owned by the session loop, never injected
into a tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_agno_v1.core.step_result import StepResult


@dataclass
class ToolEvent:
    """One member tool execution observed on the stream."""

    sop: str  # which member (delegate target) the call belongs to
    name: str
    args: dict
    result: str


@dataclass
class TurnContext:
    """Static-per-turn ids + the mutable observation accumulators."""

    user_id: str = "anonymous"
    session_id: str = "session"
    # The member currently being delegated to (set on a delegate tool_start).
    current_member: str | None = None
    tool_events: list[ToolEvent] = field(default_factory=list)
    step_results: list[StepResult] = field(default_factory=list)

    def member_tool_events(self, sop: str) -> list[ToolEvent]:
        return [e for e in self.tool_events if e.sop == sop]


__all__ = ["TurnContext", "ToolEvent"]
