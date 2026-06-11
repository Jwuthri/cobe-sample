"""A scripted fake Agno event stream + team, to drive the session without an LLM.

The fakes reproduce the exact ``.event`` taxonomy of agno 2.6.x (``TeamRunContent``
leader deltas, ``RunContent`` member deltas, ``TeamToolCallStarted/Completed`` for
delegations, ``ToolCallStarted/Completed`` for member tools) so the session's
event bridge is tested against the real contract. A ``Delegation`` may carry a
``mutate`` callback that mutates the shared cart when its tool completes, exactly
as a real member tool would.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class FakeTool:
    tool_name: str
    tool_args: dict = field(default_factory=dict)
    result: str = ""


@dataclass
class FakeEvent:
    event: str
    tool: Any = None
    content: Any = None
    content_type: str = "str"


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
    result: str = ""
    mutate: Callable[[], None] | None = None  # applied when the tool completes


@dataclass
class Delegation:
    member_id: str  # url-safe form, e.g. "product-rec"
    tools: list[ToolCall] = field(default_factory=list)
    member_reply: str = ""  # member's internal RunContent (NOT user-facing)


class FakeTeam:
    """Yields a scripted agno-shaped event stream; mutates the cart as tools run."""

    def __init__(self, delegations: list[Delegation], leader_reply: str) -> None:
        self.delegations = delegations
        self.leader_reply = leader_reply
        self.members: list[Any] = []

    def arun(self, text: str, **kwargs: Any):  # async-generator method (no await)
        return self._gen(text)

    async def _gen(self, text: str):
        yield FakeEvent("TeamRunStarted")
        for d in self.delegations:
            yield FakeEvent(
                "TeamToolCallStarted",
                tool=FakeTool("delegate_task_to_member", {"member_id": d.member_id, "task": text}),
            )
            yield FakeEvent("RunStarted")
            for tc in d.tools:
                yield FakeEvent("ToolCallStarted", tool=FakeTool(tc.name, tc.args))
                if tc.mutate is not None:
                    tc.mutate()
                yield FakeEvent("ToolCallCompleted", tool=FakeTool(tc.name, tc.args, tc.result))
            for ch in _chunks(d.member_reply):  # member chatter — must NOT be user-facing
                yield FakeEvent("RunContent", content=ch)
            yield FakeEvent("RunCompleted")
            yield FakeEvent(
                "TeamToolCallCompleted",
                tool=FakeTool("delegate_task_to_member", {"member_id": d.member_id}, "ok"),
            )
        for ch in _chunks(self.leader_reply):  # leader's user-facing reply
            yield FakeEvent("TeamRunContent", content=ch)
        yield FakeEvent("TeamRunCompleted", content=self.leader_reply)


def _chunks(text: str) -> list[str]:
    """Split into small deltas (simulate token streaming)."""
    return [w + " " for w in text.split(" ")] if text else []
