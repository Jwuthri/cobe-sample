"""Runtime — the thin, generic glue between the pure domain and the agents.

Nothing here is shopping-specific *behavior*; it is the plumbing every agent reuses:
the shared :class:`ShoppingDeps` (resolved from a registry key in ADK session state),
the :class:`StepResult` workers produce, model resolution, the SSE event vocabulary,
deep-trace frames, input guardrails, and the ``agent-as-tool`` delegation wrapper.
"""

from __future__ import annotations

from google_adk_agent_v1.runtime.delegation import Worker, deps_from, run_subagent, tool_returns
from google_adk_agent_v1.runtime.deps import ShoppingDeps
from google_adk_agent_v1.runtime.model import MODEL_NAME, gen_config, make_model
from google_adk_agent_v1.runtime.step import StepResult

__all__ = [
    "Worker",
    "run_subagent",
    "tool_returns",
    "deps_from",
    "ShoppingDeps",
    "MODEL_NAME",
    "make_model",
    "gen_config",
    "StepResult",
]
