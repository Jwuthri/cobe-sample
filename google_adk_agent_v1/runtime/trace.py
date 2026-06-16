"""Deep-trace frames — surface the *internal* traffic of a turn to the debug UI.

The normal event vocabulary (router / tool / step / token / bot) tells you *that* a
worker ran and *what* it summarized. Trace frames add the layer below: the exact
payloads moving between actors — what the orchestrator sees, what it sends into a
sub-agent, what the sub-agent hands back, and the payload the writer composes from.

All of it rides one backward-safe event: ``{type: "trace", ...}``. Production turns
set ``deps.debug = False`` and emit zero frames. Five phases are emitted:
``orchestrator_input`` and ``writer_payload`` (by the session), ``subagent_input``,
``subagent_output``, ``context`` (by the delegation wrapper).
"""

from __future__ import annotations

from typing import Any

_MAX_FIELD_CHARS = 2000  # wire guard; the frontend truncates further


def _trim(text: str) -> str:
    if len(text) > _MAX_FIELD_CHARS:
        return text[:_MAX_FIELD_CHARS] + f"… (+{len(text) - _MAX_FIELD_CHARS} more chars)"
    return text


def frame(phase: str, agent: str, title: str, data: dict) -> dict:
    """Build a ``{type: "trace"}`` SSE event."""
    return {"type": "trace", "phase": phase, "agent": agent, "title": title, "data": data}


def render_messages(events: Any) -> list[dict]:
    """Render a worker's ADK event list into compact, JSON-safe rows for the UI.

    Walks the events a sub-run produced and flattens them into role-tagged rows: the
    model's text + tool calls, and the tool results that came back.
    """
    rows: list[dict] = []
    for ev in events or []:
        content = getattr(ev, "content", None)
        if not content or not content.parts:
            continue
        for part in content.parts:
            if getattr(part, "text", None):
                rows.append({"role": "ai", "content": _trim(str(part.text))})
            fc = getattr(part, "function_call", None)
            if fc is not None:
                rows.append(
                    {"role": "ai", "content": "", "tool_calls": [{"name": fc.name, "args": dict(fc.args or {})}]}
                )
            fr = getattr(part, "function_response", None)
            if fr is not None:
                rows.append({"role": "tool", "name": fr.name, "content": _trim(str(_response_value(fr.response)))})
    return rows


def _response_value(response: Any) -> Any:
    """ADK wraps a tool's string return as ``{"result": <str>}`` — unwrap for display."""
    if isinstance(response, dict) and set(response) == {"result"}:
        return response["result"]
    return response
