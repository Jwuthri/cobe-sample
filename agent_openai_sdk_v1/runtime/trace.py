"""Deep-trace frames — surface the *internal* traffic of a turn to the debug UI.

Same envelope as :mod:`pydantic_agent_v1.runtime.trace` — a single backward-safe
``{type: "trace", ...}`` event with ``phase`` / ``agent`` / ``title`` / ``data`` —
so the existing web UI renders the panels unchanged. The only thing that changes
is the SDK-side message renderer: we walk the SDK's ``RunItem`` list instead of a
pydantic-ai message list.
"""

from __future__ import annotations

from typing import Any

from agents.items import (
    HandoffCallItem,
    ItemHelpers,
    MessageOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
)

_MAX_FIELD_CHARS = 2000  # wire guard; the frontend truncates further


def _trim(text: str) -> str:
    if len(text) > _MAX_FIELD_CHARS:
        return text[:_MAX_FIELD_CHARS] + f"… (+{len(text) - _MAX_FIELD_CHARS} more chars)"
    return text


def frame(phase: str, agent: str, title: str, data: dict) -> dict:
    """Build a ``{type: "trace"}`` SSE event."""
    return {"type": "trace", "phase": phase, "agent": agent, "title": title, "data": data}


def render_run_items(items: list[Any]) -> list[dict]:
    """Render the SDK's ``RunItem`` list into compact, JSON-safe rows for the UI.

    Each ``RunItem`` wraps an OpenAI Responses-API output item. We project the
    common cases — assistant text, tool calls, tool outputs, handoffs — and fall
    back to a generic row for anything else.
    """
    rows: list[dict] = []
    for it in items or []:
        if isinstance(it, MessageOutputItem):
            rows.append({"role": "ai", "content": _trim(ItemHelpers.text_message_output(it))})
        elif isinstance(it, ToolCallItem):
            name = it.tool_name or ""
            args = getattr(it.raw_item, "arguments", "") or ""
            rows.append({"role": "ai", "content": "", "tool_calls": [{"name": name, "args": args}]})
        elif isinstance(it, ToolCallOutputItem):
            rows.append(
                {
                    "role": "tool",
                    "name": "tool",  # name lookup happens in delegation.tool_returns when needed
                    "content": _trim(str(it.output if it.output is not None else "")),
                }
            )
        elif isinstance(it, HandoffCallItem):
            name = getattr(it.raw_item, "name", "handoff") or "handoff"
            rows.append({"role": "ai", "content": "", "tool_calls": [{"name": name, "args": ""}]})
        else:
            rows.append({"role": getattr(it, "type", "?"), "content": _trim(str(getattr(it, "raw_item", "")))})
    return rows
