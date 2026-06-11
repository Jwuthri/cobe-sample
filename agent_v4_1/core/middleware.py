"""Built-in platform middleware factories, registered by name.

Each factory takes the ``MiddlewareSpec.params`` as kwargs and returns an
``AgentMiddleware``. The config references them by name (e.g.
``{"name": "max_turns", "params": {"max_turns": 30}}``).
"""

from __future__ import annotations

from typing import Any, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.config import get_stream_writer
from langgraph.types import Command

from agent_v4_1.core.registry import MIDDLEWARE


# =============================================================================
# model_call_counter — observability (counts model calls on the instance)
# =============================================================================
class _ModelCallCounter(AgentMiddleware):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def before_model(self, state: Any, runtime: Any) -> None:
        self.count += 1
        return None


def model_call_counter() -> AgentMiddleware:
    return _ModelCallCounter()


# =============================================================================
# max_turns — cap model calls (v4's MAX_ITERATIONS analogue)
# =============================================================================
def max_turns(max_turns: int = 30) -> AgentMiddleware:
    """Cap model calls per run. Maps to ``ModelCallLimitMiddleware``.

    Note: without a checkpointer the run is the whole session for our per-session
    agents, so this is effectively a per-turn model-call budget.
    """
    return ModelCallLimitMiddleware(run_limit=max_turns, exit_behavior="end")


# =============================================================================
# tool_call_limit — cap tool calls (the orchestrator's loop bound)
# =============================================================================
def tool_call_limit(run_limit: int = 4, exit_behavior: str = "end") -> AgentMiddleware:
    return ToolCallLimitMiddleware(run_limit=run_limit, exit_behavior=exit_behavior)


# =============================================================================
# log_tool_calls — emit custom stream events for the SSE/CLI observability layer
# =============================================================================
def _safe_stream_writer():
    try:
        return get_stream_writer()
    except Exception:
        return None


class LogToolCallsMiddleware(AgentMiddleware):
    """Emit tool_start/tool_end custom events around each tool call.

    Implements BOTH sync and async variants: a sub-agent runs sync (via the
    re-pump ``stream``), but the orchestrator runs under ``astream`` and would
    otherwise hit "awrap_tool_call not available".
    """

    def __init__(self, log_prefix: str = "") -> None:
        super().__init__()
        self.log_prefix = log_prefix

    def _emit_start(self, request: ToolCallRequest) -> None:
        writer = _safe_stream_writer()
        if writer is not None:
            writer(
                {
                    "event": "tool_start",
                    "tool": request.tool_call["name"],
                    "args": request.tool_call.get("args", {}),
                    "agent": self.log_prefix,
                }
            )

    def _emit_end(self, request: ToolCallRequest, result: ToolMessage | Command) -> None:
        writer = _safe_stream_writer()
        if writer is None:
            return
        content = ""
        if isinstance(result, ToolMessage):
            content = str(result.content)
        elif isinstance(result, Command):
            msgs = (getattr(result, "update", None) or {}).get("messages")
            if msgs:
                content = str(msgs[-1].content)
        if len(content) > 400:
            content = content[:400] + "…"
        writer(
            {
                "event": "tool_end",
                "tool": request.tool_call["name"],
                "result": content,
                "agent": self.log_prefix,
            }
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        self._emit_start(request)
        result = handler(request)
        self._emit_end(request, result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> ToolMessage | Command:
        self._emit_start(request)
        result = await handler(request)
        self._emit_end(request, result)
        return result


def log_tool_calls(log_prefix: str = "") -> AgentMiddleware:
    return LogToolCallsMiddleware(log_prefix)


def register_builtin_middleware() -> None:
    """Idempotently register the built-in middleware factories."""
    for name, factory in (
        ("model_call_counter", model_call_counter),
        ("max_turns", max_turns),
        ("tool_call_limit", tool_call_limit),
        ("log_tool_calls", log_tool_calls),
    ):
        if not MIDDLEWARE.has(name):
            MIDDLEWARE.register(name, factory)


__all__ = [
    "model_call_counter",
    "max_turns",
    "tool_call_limit",
    "log_tool_calls",
    "LogToolCallsMiddleware",
    "register_builtin_middleware",
]
