"""agent_v4_1 — config-driven sub-agents on ``create_agent``, streaming-first.

A cleaner rebuild of agent_v4's declarative idea on the v5 agent-as-tool topology:

  * ``core/``     — the reusable platform: the :class:`AgentConfig` contract,
                    :func:`build_agent`, the registries, and the generic
                    sub-agent-as-tool wrapper. Tenant-agnostic.
  * ``shopping/`` — the demo tenant: domain, tools, the three sub-agents, the
                    streaming orchestrator + writer session.

The headline change vs v4/v5 is **true token streaming**: the writer is the last
model call in a turn with nothing after it, so its tokens stream straight to the
client (see ``agent_v4_1/README.md`` for the architecture).
"""

from __future__ import annotations

try:  # load .env like the rest of the repo (no hard dep if python-dotenv is absent)
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

from agent_v4_1.core import (  # noqa: E402
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
