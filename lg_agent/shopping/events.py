"""Map orchestrator stream chunks → the frontend's SSE event vocabulary.

The orchestrator is streamed with ``stream_mode=["updates", "custom"]``. The custom
events come from ``log_tool_calls`` (a sub-agent call → ``router`` + ``agent`` rows;
an inner tool/skill → ``tool_start`` / ``tool_end`` / ``skill``) and from deep-trace
frames. The session drains ``ctx.step_results`` for ``step`` rows; token streaming
and the final ``bot`` event are produced separately by the session.
"""

from __future__ import annotations

from typing import Any

from lg_agent.shopping.agents.subagents.names import ALL as _SUBAGENT_NAMES_TUPLE

# A tool_start on one of these names is a routing event (a sub-agent call).
_SUBAGENT_NAMES = set(_SUBAGENT_NAMES_TUPLE)


def classify_custom(payload: Any) -> list[dict]:
    """Turn one custom stream chunk into 0+ UI events."""
    if not isinstance(payload, dict):
        return []
    ev = payload.get("event")
    name = payload.get("tool")
    if ev == "trace":  # a deep-trace frame emitted from inside a sub-agent tool
        tr = payload.get("trace")
        return [tr] if isinstance(tr, dict) else []
    if ev == "tool_start":
        args = {k: v for k, v in (payload.get("args") or {}).items() if k != "tool_call_id"}
        if name in _SUBAGENT_NAMES:
            return [{"type": "router", "target": name, "iteration": 0}]
        if name == "load_skill":
            return [{"type": "skill", "name": args.get("skill_name")}]
        return [{"type": "tool_start", "name": name, "args": args}]
    if ev == "tool_end":
        if name in _SUBAGENT_NAMES:
            return [{"type": "agent", "node": f"{name}_wrapper"}]
        return [{"type": "tool_end", "name": name, "result": payload.get("result", "")}]
    return []


def is_subagent_tool_end(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("event") == "tool_end"
        and payload.get("tool") in _SUBAGENT_NAMES
    )


def step_event(sr) -> dict:
    return {
        "type": "step",
        "sop": sr.sop,
        "summary": sr.summary,
        "asks": list(sr.asks),
        "next_sop": sr.next_sop,
        "details": sr.details,
    }


__all__ = ["classify_custom", "is_subagent_tool_end", "step_event"]
