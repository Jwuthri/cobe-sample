"""Deep-trace observability — render the *internal* traffic of a turn for the UI.

The base SSE vocabulary (router / agent / tool / step / token / bot) tells you
*that* a sub-agent ran and *what* it summarized. This module adds the layer below:
the exact payloads moving between the actors —

  * what the orchestrator sees of the conversation + its own system prompt;
  * what the orchestrator sends *into* a sub-agent (its prompt, tools, and the
    self-contained ``query`` it actually receives);
  * what the sub-agent hands *back* (its raw messages, the distilled StepResult,
    and the terse string the orchestrator LLM actually reads);
  * the runtime ``TurnContext`` after each sub-agent mutates it;
  * the exact JSON payload the writer composes its reply from.

All of it rides one backward-safe SSE event: ``{type:"trace", ...}``. Session code
yields these directly; sub-agent tools (which run *inside* the orchestrator run)
emit them on the context's event bus via :func:`emit_trace`. Everything is
pre-trimmed and JSON-safe so it survives ``json.dumps`` straight onto the wire.
"""

from __future__ import annotations

import json
from typing import Any

# A single per-field cap so one giant message can't blow up the SSE frame; the
# frontend collapses + truncates further, this is just the wire guard.
_MAX_FIELD_CHARS = 2000


def _trim(text: str, max_chars: int = _MAX_FIELD_CHARS) -> str:
    if len(text) > max_chars:
        return text[:max_chars] + f"… (+{len(text) - max_chars} more chars)"
    return text


def render_message(message: Any, max_chars: int = _MAX_FIELD_CHARS) -> dict:
    """One ``Msg`` → a compact, JSON-safe dict (role / text / tool calls)."""
    role = getattr(message, "role", None) or message.__class__.__name__
    out: dict[str, Any] = {"role": role, "content": _trim(str(getattr(message, "content", "")), max_chars)}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {"name": tc.get("name"), "args": tc.get("args", {})} for tc in tool_calls
        ]
    name = getattr(message, "name", None)
    if name:
        out["name"] = name
    return out


def render_messages(messages: Any, max_chars: int = _MAX_FIELD_CHARS) -> list[dict]:
    return [render_message(m, max_chars) for m in (messages or [])]


def trace_event(phase: str, agent: str, title: str, data: dict) -> dict:
    """Build a ``{type:"trace"}`` UI event (yield this straight from session code)."""
    return {"type": "trace", "phase": phase, "agent": agent, "title": title, "data": data}


def emit_trace(ctx: Any, phase: str, agent: str, title: str, data: dict) -> None:
    """Emit a trace from *inside* the orchestrator run (a sub-agent tool body).

    Pushes a ``{type:"trace"}`` event onto the context's live event bus so it
    reaches the session's turn loop. No-op when there is no bus (e.g. a plain
    ``Runner.run`` in a test).
    """
    if ctx is not None and getattr(ctx, "bus", None) is not None:
        ctx.emit(trace_event(phase, agent, title, data))


def to_jsonable(value: Any) -> Any:
    """Best-effort JSON-safe coercion (Decimals, pydantic, dataclasses, sets)."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return json.loads(json.dumps(value, default=str))


__all__ = [
    "render_message",
    "render_messages",
    "trace_event",
    "emit_trace",
    "to_jsonable",
]
