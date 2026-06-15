"""Runtime — the thin, generic glue between the pure domain and the agents.

Nothing here is shopping-specific *behavior*; it is the plumbing every agent reuses:
the shared :class:`ShoppingDeps`, the :class:`StepResult` workers produce, model
resolution, the SSE event vocabulary, deep-trace frames, input guardrails, and the
``agent-as-tool`` delegation wrapper.
"""

from __future__ import annotations

from pydantic_agent_v1.runtime.delegation import Worker, run_subagent, tool_returns
from pydantic_agent_v1.runtime.deps import ShoppingDeps
from pydantic_agent_v1.runtime.model import MODEL_NAME, settings
from pydantic_agent_v1.runtime.step import StepResult

__all__ = [
    "Worker",
    "run_subagent",
    "tool_returns",
    "ShoppingDeps",
    "MODEL_NAME",
    "settings",
    "StepResult",
]
