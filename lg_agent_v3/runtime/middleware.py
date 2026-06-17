"""Middleware primitives — the harness features that replace Pydantic AI natives.

Pydantic AI gives a few things as first-class features; LangChain expresses the same
ideas as ``AgentMiddleware`` that hooks ``wrap_model_call``. Three generic primitives
live here so the agents can declare their behavior in one line each:

  * :func:`dynamic_instructions` — Pydantic AI's ``@agent.instructions``: a transient
    ``SystemMessage`` computed from the live shared state, re-rendered on EVERY model
    call (so a mid-turn cart mutation updates it for free). Used for the orchestrator's
    routing memo, product_rec's cart note, and checkout's progress anchor.
  * :func:`hide_tool` — Pydantic AI's ``Tool(prepare=...)`` native tool-gating: drop a
    tool from the model's options while a predicate holds (the empty-cart guard).
  * :func:`no_parallel_tools` — Pydantic AI's ``parallel_tool_calls=False``: route ONE
    tool per model step (keeps the bus ordered + the shared cart race-free). Forwarded
    to ``bind_tools`` via the request's ``model_settings``.

Each implements BOTH ``wrap_model_call`` and ``awrap_model_call`` (the orchestrator
runs async; a worker runs via a nested ``ainvoke``).
"""

from __future__ import annotations

from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage


def _context(request: Any) -> Any:
    return getattr(getattr(request, "runtime", None), "context", None)


class _DynamicInstructions(AgentMiddleware):
    """Prepend a transient SystemMessage rendered from the shared deps each call."""

    def __init__(self, render: Callable[[Any], str]) -> None:
        super().__init__()
        self._render = render

    def _apply(self, request: Any) -> Any:
        ctx = _context(request)
        if ctx is None:
            return request
        note = self._render(ctx)
        if note:
            # APPEND the volatile note at the very end, not the front: the prefix
            # (system + tools + history) then stays stable + cacheable turn-to-turn.
            # It must go AFTER the last message, never between one — the model node
            # always runs with the last message a Human (turn start) or a ToolMessage
            # (mid tool-loop), and a SystemMessage inserted *before* a trailing
            # ToolMessage splits it from its AIMessage(tool_calls) → provider 400.
            # A trailing system reminder is valid in both cases. ponytail: append, not splice.
            request = request.override(messages=[*request.messages, SystemMessage(content=note)])
        return request

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(self._apply(request))


class _HideTool(AgentMiddleware):
    """Strip a tool from the model's options while ``predicate(deps)`` is True."""

    def __init__(self, tool_name: str, predicate: Callable[[Any], bool]) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._predicate = predicate

    def _apply(self, request: Any) -> Any:
        ctx = _context(request)
        if ctx is not None and request.tools and self._predicate(ctx):
            kept = [t for t in request.tools if getattr(t, "name", None) != self._tool_name]
            if len(kept) != len(request.tools):
                request = request.override(tools=kept)
        return request

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(self._apply(request))


class _ModelSettings(AgentMiddleware):
    """Merge fixed ``model_settings`` into every model request (→ ``bind_tools``)."""

    def __init__(self, **settings: Any) -> None:
        super().__init__()
        self._settings = settings

    def _apply(self, request: Any) -> Any:
        return request.override(model_settings={**(request.model_settings or {}), **self._settings})

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(self._apply(request))


def dynamic_instructions(render: Callable[[Any], str]) -> AgentMiddleware:
    """Inject a transient SystemMessage rendered from deps on every model call."""
    return _DynamicInstructions(render)


def hide_tool(tool_name: str, predicate: Callable[[Any], bool]) -> AgentMiddleware:
    """Drop ``tool_name`` from the model's tools while ``predicate(deps)`` holds."""
    return _HideTool(tool_name, predicate)


def no_parallel_tools() -> AgentMiddleware:
    """Force one tool call per model step (``parallel_tool_calls=False``)."""
    return _ModelSettings(parallel_tool_calls=False)


__all__ = ["dynamic_instructions", "hide_tool", "no_parallel_tools"]
