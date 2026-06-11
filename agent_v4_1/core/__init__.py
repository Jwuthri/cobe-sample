"""agent_v4_1 core — the reusable, tenant-agnostic agent platform.

Importing this package self-registers the built-in middleware + guardrail
factories into the registries, so a config referencing ``log_tool_calls`` /
``pii`` / etc. resolves without any extra wiring. ``core`` never imports
``shopping`` — the dependency runs one way only.
"""

from __future__ import annotations

from agent_v4_1.core.config import AgentConfig
from agent_v4_1.core.factory import build_agent
from agent_v4_1.core.guardrails import register_builtin_guardrails
from agent_v4_1.core.middleware import register_builtin_middleware
from agent_v4_1.core.registry import GUARDRAILS, MIDDLEWARE, SKILLS, TOOLS, register_tool

register_builtin_middleware()
register_builtin_guardrails()

__all__ = [
    "AgentConfig",
    "build_agent",
    "register_tool",
    "TOOLS",
    "SKILLS",
    "MIDDLEWARE",
    "GUARDRAILS",
]
