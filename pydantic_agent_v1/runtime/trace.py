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


def render_messages(messages: Any) -> list[dict]:
    """Render a Pydantic AI message list into compact, JSON-safe rows for the UI."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    rows: list[dict] = []
    for m in messages or []:
        if isinstance(m, ModelRequest):
            for part in m.parts:
                if isinstance(part, SystemPromptPart):
                    rows.append({"role": "system", "content": _trim(str(part.content))})
                elif isinstance(part, UserPromptPart):
                    rows.append({"role": "human", "content": _trim(str(part.content))})
                elif isinstance(part, ToolReturnPart):
                    rows.append(
                        {"role": "tool", "name": part.tool_name, "content": _trim(str(part.content))}
                    )
        elif isinstance(m, ModelResponse):
            for part in m.parts:
                if isinstance(part, TextPart):
                    rows.append({"role": "ai", "content": _trim(str(part.content))})
                elif isinstance(part, ToolCallPart):
                    rows.append(
                        {"role": "ai", "content": "", "tool_calls": [{"name": part.tool_name, "args": part.args}]}
                    )
    return rows
