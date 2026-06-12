"""openai_agent_v1 — config-driven sub-agents on the OpenAI Agents SDK, streaming-first.

A clean-room rebuild of agent_v4_1's declarative idea (and its agent-as-tool
topology) on the **OpenAI Agents SDK** instead of LangChain ``create_agent`` —
imports nothing from agent_v4_1.

  * ``core/``     — the reusable platform: the :class:`AgentConfig` contract,
                    :func:`build_agent` (→ ``agents.Agent``), the registries, and
                    the generic sub-agent-as-tool wrapper. Tenant-agnostic.
  * ``shopping/`` — the demo tenant: domain, tools, the three sub-agents, the
                    streaming orchestrator + writer session.

True token streaming is preserved: the writer is the last model call in a turn
with nothing after it, so its tokens stream straight to the client (see
``openai_agent_v1/README.md``).
"""

from __future__ import annotations

try:  # load .env like the rest of the repo (no hard dep if python-dotenv is absent)
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

from openai_agent_v1.core import (  # noqa: E402
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
