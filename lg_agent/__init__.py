"""``lg_agent`` — a clean, config-driven multi-agent shopping assistant.

Two layers, cleanly separated:

  * :mod:`lg_agent.core`     — the reusable, tenant-agnostic platform: the
    ``AgentConfig`` JSON contract, the capability registries, ``build_agent``, and
    the sub-agent-as-tool wrapper. It never imports a tenant.
  * :mod:`lg_agent.shopping` — the demo tenant: the e-commerce domain, the leaf
    tools, the two pre-defined agents (orchestrator + writer), the three on-the-fly
    sub-agents, and the streaming turn engine.

The headline trait is **true token streaming**: the writer is the last model call
of a turn with nothing after it, so its tokens stream straight to the client. See
``lg_agent/README.md`` for the full architecture.
"""

from __future__ import annotations

try:  # load .env like the rest of the repo (no hard dep if python-dotenv is absent)
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

from lg_agent.core import (  # noqa: E402
    GUARDRAILS,
    MIDDLEWARE,
    SKILLS,
    TOOLS,
    AgentConfig,
    build_agent,
    register_tool,
)

__all__ = [
    "AgentConfig",
    "build_agent",
    "register_tool",
    "TOOLS",
    "SKILLS",
    "MIDDLEWARE",
    "GUARDRAILS",
]
