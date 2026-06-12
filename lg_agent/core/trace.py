"""Deep-trace observability — surface the *internal* traffic of a turn to the UI.

The normal SSE vocabulary (router / agent / tool / step / token / bot) tells you
*that* a sub-agent ran and *what* it summarized. Trace frames add the layer below:
the exact payloads moving between actors — what the orchestrator sees, what it
sends into a sub-agent, what the sub-agent hands back, and the payload the writer
composes from.

All of it rides one backward-safe SSE event: ``{type:"trace", ...}``.

Two emission paths:
  * the session yields frames directly (it runs *outside* the orchestrator graph) —
    use :func:`frame`;
  * a sub-agent tool runs *inside* the graph, so it emits on the custom stream via
    :func:`emit`, and ``shopping.events.classify_custom`` lifts the frame back out.

Both paths are debug-gated by the caller / ``ctx.debug`` so production turns emit
zero frames.
"""

from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer

# Per-field cap so one giant message can't blow up an SSE frame (the frontend
# truncates further; this is just the wire guard).
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
                else:  # tool_use / image / ... — a compact marker, not the blob
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
    out: dict[str, Any] = {
        "role": role,
        "content": _trim(_content_to_text(getattr(message, "content", "")), max_chars),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [{"name": tc.get("name"), "args": tc.get("args", {})} for tc in tool_calls]
    name = getattr(message, "name", None)
    if name:
        out["name"] = name
    return out


def render_messages(messages: Any, max_chars: int = _MAX_FIELD_CHARS) -> list[dict]:
    return [render_message(m, max_chars) for m in (messages or [])]


def frame(phase: str, agent: str, title: str, data: dict) -> dict:
    """Build a ``{type:"trace"}`` UI event (yield this straight from session code)."""
    return {"type": "trace", "phase": phase, "agent": agent, "title": title, "data": data}


def emit(phase: str, agent: str, title: str, data: dict) -> None:
    """Emit a trace from *inside* the orchestrator graph (a sub-agent tool body).

    Wrapped as a custom stream chunk so it reaches the orchestrator's ``astream``
    consumer. No-op when there is no active stream writer (e.g. a plain ``.invoke``
    in a test).
    """
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None
    if writer is not None:
        writer({"event": "trace", "trace": frame(phase, agent, title, data)})


__all__ = ["render_message", "render_messages", "frame", "emit"]
