"""Runtime — the thin, generic glue between the pure domain and the agents.

Nothing here is shopping-specific *behavior*; it is the plumbing every agent reuses:
the shared :class:`ShoppingDeps`, the :class:`StepResult` workers produce, model
resolution, the middleware primitives that replace Pydantic AI natives, the SSE event
vocabulary, deep-trace frames, input guardrails, and the ``agent-as-tool`` delegation
wrapper.
"""

from __future__ import annotations

from lg_agent_v2.runtime.delegation import ToolReturn, Worker, run_subagent, tool_returns
from lg_agent_v2.runtime.deps import ShoppingDeps
from lg_agent_v2.runtime.middleware import dynamic_instructions, hide_tool, no_parallel_tools
from lg_agent_v2.runtime.model import MODEL_NAME, build_model
from lg_agent_v2.runtime.step import StepResult

__all__ = [
    "Worker",
    "run_subagent",
    "tool_returns",
    "ToolReturn",
    "ShoppingDeps",
    "MODEL_NAME",
    "build_model",
    "dynamic_instructions",
    "hide_tool",
    "no_parallel_tools",
    "StepResult",
]
