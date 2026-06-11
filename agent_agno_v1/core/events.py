"""Map Agno stream events → the frontend's SSE event vocabulary.

The team is run with ``arun(stream=True, stream_events=True)``. Agno yields a flat
stream that interleaves **leader** events (``"Team…"``-prefixed ``.event`` values)
and forwarded **member** events (plain agent ``.event`` values). The session loop
discriminates purely on the ``.event`` string — robust against the fact that the
leader and member event classes share Python class names:

  * ``TeamRunContent``    → the leader's user-facing reply, streamed as ``token``;
  * ``RunContent``        → a member's internal chatter (NOT user-facing — dropped);
  * ``TeamToolCallStarted`` w/ a delegate tool → a ``router`` event (target = member id);
  * ``ToolCallStarted/Completed`` (member domain tools) → ``tool_start`` / ``tool_end``;
  * ``TeamRunCompleted``  → the authoritative final text.

This module is pure helpers; the stateful orchestration (tracking the current
member, distilling ``StepResult`` s, emitting ``step`` rows) lives in the session.
"""

from __future__ import annotations

from typing import Any

# ---- the .event string discriminators (verified against agno 2.6.13 source) ----
LEADER_TOKEN = "TeamRunContent"
MEMBER_TOKEN = "RunContent"
LEADER_TOOL_START = "TeamToolCallStarted"
LEADER_TOOL_END = "TeamToolCallCompleted"
MEMBER_TOOL_START = "ToolCallStarted"
MEMBER_TOOL_END = "ToolCallCompleted"
TEAM_RUN_COMPLETED = "TeamRunCompleted"
RUN_COMPLETED = "RunCompleted"

# The leader's hand-off tools (a TeamToolCallStarted on one of these is routing).
DELEGATE_TOOLS = {"delegate_task_to_member", "delegate_task_to_members"}


def ev_name(ev: Any) -> str:
    """The ``.event`` string of an Agno run event (works for object or dict)."""
    if isinstance(ev, dict):
        return str(ev.get("event", ""))
    return str(getattr(ev, "event", "") or "")


def ev_tool(ev: Any) -> Any:
    """The ``ToolExecution`` carried by a tool-call event (``.tool``), or None."""
    if isinstance(ev, dict):
        return ev.get("tool")
    return getattr(ev, "tool", None)


def ev_content(ev: Any) -> str:
    """The text delta carried by a content event, when it is plain string content."""
    content_type = ev.get("content_type") if isinstance(ev, dict) else getattr(ev, "content_type", "str")
    if content_type not in (None, "str"):
        return ""
    content = ev.get("content") if isinstance(ev, dict) else getattr(ev, "content", None)
    return content if isinstance(content, str) else ""


def tool_name(tool: Any) -> str:
    return str(getattr(tool, "tool_name", "") or "")


def tool_args(tool: Any) -> dict:
    args = getattr(tool, "tool_args", None) or {}
    return args if isinstance(args, dict) else {}


def tool_result(tool: Any) -> str:
    return str(getattr(tool, "result", "") or "")


def delegate_target(tool: Any) -> str | None:
    """The member id a delegate tool is handing off to (from its args)."""
    return tool_args(tool).get("member_id") or tool_args(tool).get("member_name")


def canonical_member(raw: str | None) -> str:
    """Normalise Agno's url-safe member id back to the internal sop vocabulary.

    Agno presents member ids in url-safe (kebab) form — ``"product-rec"``,
    ``"order-status"`` — while the sop vocabulary (StepResult.sop / BLOCK_BY_SOP /
    EXTRACTORS) uses snake_case. Map kebab → snake so lookups line up.
    """
    return (raw or "").replace("-", "_").lower()


def is_member_not_found(result: str) -> bool:
    """True when a delegate tool result is the 'member not found' error string."""
    low = (result or "").lower()
    return "not found" in low and "member" in low


# ---- SSE event constructors (the frontend's vocabulary) ----
def token_event(delta: str) -> dict:
    return {"type": "token", "content": delta}


def router_event(target: str) -> dict:
    return {"type": "router", "target": target, "iteration": 0}


def agent_event(member_id: str) -> dict:
    return {"type": "agent", "node": f"{member_id}_wrapper"}


def tool_start_event(tool: Any) -> dict:
    return {"type": "tool_start", "name": tool_name(tool), "args": tool_args(tool)}


def tool_end_event(tool: Any) -> dict:
    result = tool_result(tool)
    if len(result) > 600:
        result = result[:600] + "…"
    return {"type": "tool_end", "name": tool_name(tool), "result": result}


def step_event(sr: Any) -> dict:
    return {
        "type": "step",
        "sop": sr.sop,
        "summary": sr.summary,
        "asks": list(sr.asks),
        "next_sop": sr.next_sop,
        "details": sr.details,
    }


__all__ = [
    "LEADER_TOKEN",
    "MEMBER_TOKEN",
    "LEADER_TOOL_START",
    "LEADER_TOOL_END",
    "MEMBER_TOOL_START",
    "MEMBER_TOOL_END",
    "TEAM_RUN_COMPLETED",
    "RUN_COMPLETED",
    "DELEGATE_TOOLS",
    "ev_name",
    "ev_tool",
    "ev_content",
    "tool_name",
    "tool_args",
    "tool_result",
    "delegate_target",
    "canonical_member",
    "is_member_not_found",
    "token_event",
    "router_event",
    "agent_event",
    "tool_start_event",
    "tool_end_event",
    "step_event",
]
