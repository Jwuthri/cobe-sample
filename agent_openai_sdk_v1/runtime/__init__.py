"""Runtime — the thin, generic glue between the pure domain and the agents.

Nothing here is shopping-specific *behavior*; it is the plumbing every agent reuses:
the shared :class:`ShoppingContext`, the :class:`StepResult` workers produce, model
resolution, the SSE event vocabulary, deep-trace frames, input guardrails, and the
``agent-as-tool`` wrapper.
"""

from __future__ import annotations

from agent_openai_sdk_v1.runtime.context import ShoppingContext
from agent_openai_sdk_v1.runtime.delegation import Worker, build_worker_tool, tool_returns
from agent_openai_sdk_v1.runtime.model import MODEL_NAME, settings
from agent_openai_sdk_v1.runtime.step import StepResult

__all__ = [
    "ShoppingContext",
    "Worker",
    "build_worker_tool",
    "tool_returns",
    "MODEL_NAME",
    "settings",
    "StepResult",
]
