"""Deep-trace observability — render the *internal* traffic of a turn for the UI.

The existing SSE vocabulary (router / agent / tool / step / token / bot) tells you
*that* a sub-agent ran and *what* it summarized. This module adds the layer below:
the exact payloads moving between the actors —

  * what the orchestrator sees of the conversation + its own system prompt;
  * what the orchestrator sends *into* a sub-agent (its prompt, tools, and the
    bounded conversation window it actually receives), plus the ``query``;
  * what the sub-agent hands *back* (its raw messages, the distilled StepResult,
    and the terse string the orchestrator LLM actually reads);
  * the runtime ``TurnContext`` after each sub-agent mutates it (step_results,
    usage, cart);
  * the exact JSON payload the writer composes its reply from.

All of it rides one new backward-safe SSE event: ``{type:"trace", ...}``. Session
code yields these directly; sub-agent tools (which run *inside* the orchestrator
graph) emit them on the custom stream via :func:`emit_trace`, and
``events.classify_custom`` lifts them back out. Everything here is pre-trimmed and
JSON-safe so it survives ``json.dumps`` straight onto the wire.
"""

from __future__ import annotations

import json
from typing import Any

from langgraph.config import get_stream_writer

# A single per-field cap so one giant message can't blow up the SSE frame; the
# frontend collapses + truncates further, this is just the wire guard.
_MAX_FIELD_CHARS = 2000


def _content_to_text(content: Any) -> str:
    """Flatten a message's ``content`` (str or content-block list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                else:  # tool_use / image / ... — show a compact marker, not the blob
                    parts.append(f"[{part.get('type', 'block')}]")
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _trim(text: str, max_chars: int = _MAX_FIELD_CHARS) -> str:
    if len(text) > max_chars:
        return text[:max_chars] + f"… (+{len(text) - max_chars} more chars)"
    return text


def render_message(message: Any, max_chars: int = _MAX_FIELD_CHARS) -> dict:
    """One message → a compact, JSON-safe dict (role / text / tool calls)."""
    role = getattr(message, "type", None) or message.__class__.__name__
    out: dict[str, Any] = {"role": role, "content": _trim(_content_to_text(getattr(message, "content", "")), max_chars)}
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


def _safe_stream_writer():
    try:
        return get_stream_writer()
    except Exception:
        return None


def emit_trace(phase: str, agent: str, title: str, data: dict) -> None:
    """Emit a trace from *inside* the orchestrator graph (a sub-agent tool body).

    Wrapped as a custom stream chunk so it reaches the orchestrator's ``astream``
    consumer; ``events.classify_custom`` unwraps it back into the UI event. No-op
    when there is no active stream writer (e.g. a plain ``.invoke`` in a test).
    """
    writer = _safe_stream_writer()
    if writer is not None:
        writer({"event": "trace", "trace": trace_event(phase, agent, title, data)})


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
