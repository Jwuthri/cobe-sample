"""Tool-call logger middleware.

Emits a custom stream event for every tool call so the rich TUI can
display the call/result. Pure observability — no behavior change.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from agent_v2 import debug_log
from langchain.agents.middleware import wrap_tool_call
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.config import get_stream_writer
from langgraph.types import Command


@wrap_tool_call
def log_tool_calls(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Record the tool name, args, and (truncated) result via a custom stream event."""
    writer = None
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None

    name = request.tool_call["name"]
    args = request.tool_call.get("args", {})
    debug_log.tool_start(name, args)
    if writer is not None:
        writer({"event": "tool_start", "tool": name, "args": args})

    result = handler(request)

    if writer is not None:
        # Truncate large results for readability.
        content = ""
        if isinstance(result, ToolMessage):
            content = str(result.content)
        elif isinstance(result, Command):
            # Command may carry a ToolMessage in its update.messages
            msgs = (result.update or {}).get("messages") if hasattr(result, "update") else None
            if msgs:
                content = str(msgs[-1].content)
        if len(content) > 400:
            content = content[:400] + "…"
        debug_log.tool_end(name, content)
        writer({"event": "tool_end", "tool": name, "result": content})
    else:
        debug_log.tool_end(name, "")
    return result
