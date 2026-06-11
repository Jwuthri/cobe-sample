"""Synthesize the frontend's SSE event vocabulary from an Agno team run.

The team is awaited (not token-streamed) for its routing phase, then we replay
each member's work as the same ``router`` / ``tool_start`` / ``tool_end`` /
``agent`` / ``step`` events the web UI already understands. The writer's tokens
ARE streamed live (see ``session.py``); only the orchestration phase is replayed
post-hoc — the final UI state is identical.
"""

from __future__ import annotations

from typing import Any


def member_events(name: str, tools: list[Any]) -> list[dict]:
    """Events for one member's contribution: route in, its tool calls, route out."""
    out: list[dict] = [{"type": "router", "target": name, "iteration": 0}]
    for te in tools or []:
        args = dict(getattr(te, "tool_args", None) or {})
        out.append({"type": "tool_start", "name": te.tool_name, "args": args})
        out.append({"type": "tool_end", "name": te.tool_name, "result": str(te.result)})
    out.append({"type": "agent", "node": f"{name}_wrapper"})
    return out


def step_event(sr) -> dict:
    return {
        "type": "step",
        "sop": sr.sop,
        "summary": sr.summary,
        "asks": list(sr.asks),
        "next_sop": sr.next_sop,
        "details": sr.details,
    }


__all__ = ["member_events", "step_event"]
