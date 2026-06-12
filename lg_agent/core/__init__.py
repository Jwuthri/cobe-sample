"""``lg_agent.core`` — the reusable, tenant-agnostic agent platform.

What lives here:

  * :mod:`config`     — the ``AgentConfig`` JSON contract for an on-the-fly agent.
  * :mod:`registry`   — the four capability registries (tools/skills/guardrails/
                        middleware) a config references by name.
  * :mod:`builder`    — ``build_agent(config)`` → a ``create_agent`` graph.
  * :mod:`subagent`   — ``SubAgent`` + ``as_tool`` (wrap a config-built agent as an
                        orchestrator tool).
  * :mod:`model` / :mod:`tools` / :mod:`skills` / :mod:`guardrails` /
    :mod:`middleware` — the resolvers behind each config field.
  * :mod:`context` / :mod:`step` / :mod:`trace` — per-turn state + observability.

``core`` never imports a tenant package; wiring always flows tenant → core.
"""

from __future__ import annotations

from lg_agent.core.builder import build_agent
from lg_agent.core.config import AgentConfig, to_config
from lg_agent.core.context import TurnContext
from lg_agent.core.guardrails import register_builtin_guardrails
from lg_agent.core.middleware import register_builtin_middleware
from lg_agent.core.registry import GUARDRAILS, MIDDLEWARE, SKILLS, TOOLS, register_tool
from lg_agent.core.step import StepResult
from lg_agent.core.subagent import SubAgent, as_tool, build_delegate_tools


def register_builtins() -> None:
    """Register the platform's built-in guardrails + middleware (idempotent)."""
    register_builtin_guardrails()
    register_builtin_middleware()


__all__ = [
    "AgentConfig",
    "to_config",
    "build_agent",
    "TurnContext",
    "StepResult",
    "SubAgent",
    "as_tool",
    "build_delegate_tools",
    "register_tool",
    "register_builtins",
    "TOOLS",
    "SKILLS",
    "MIDDLEWARE",
    "GUARDRAILS",
]
