"""SSE event helpers + the deep-trace frame.

The session yields a small, stable event vocabulary (the same the v2/v4 web UI
speaks): ``user / state / router / tool_start / tool_end / agent / step / token /
writer / bot / end`` plus two backward-safe additions — ``guardrail`` and
``trace``. This module just constructs those dicts; the orchestration lives in
:mod:`agno_agent_v1.agent.session`.
"""

from __future__ import annotations

from typing import Any

from agno_agent_v1.agent.context import StepResult

# Trim long fields so a trace frame never bloats the SSE stream.
_MAX_FIELD_CHARS = 4000


def step_event(sr: StepResult) -> dict:
    return {
        "type": "step",
        "sop": sr.sop,
        "summary": sr.summary,
        "asks": list(sr.asks),
        "next_sop": sr.next_sop,
        "details": sr.details,
    }


def _trim(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_FIELD_CHARS:
        return value[:_MAX_FIELD_CHARS] + f"… (+{len(value) - _MAX_FIELD_CHARS} chars)"
    if isinstance(value, dict):
        return {k: _trim(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_trim(v) for v in value]
    return value


def trace_event(phase: str, agent: str, title: str, data: dict) -> dict:
    """A backward-safe ``{type:"trace"}`` frame exposing the internal traffic."""
    return {"type": "trace", "phase": phase, "agent": agent, "title": title, "data": _trim(data)}


def render_messages(messages: list[Any]) -> list[dict]:
    """Render Agno ``Message`` objects (or plain dicts) as role-chipped rows."""
    out: list[dict] = []
    for m in messages or []:
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "?")
        content = getattr(m, "content", None)
        if content is None and isinstance(m, dict):
            content = m.get("content", "")
        out.append({"role": role, "content": _trim(str(content))})
    return out


__all__ = ["step_event", "trace_event", "render_messages"]
