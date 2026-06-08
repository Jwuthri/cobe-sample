"""agent_v5 — the "agent-as-tool" rebuild of agent_v4.

Same leaves, tools, cart, skills, and guardrails as v4 — but the topology is
inverted. Instead of a hand-written LangGraph ``supervisor`` node that routes via
``Command(goto="<leaf>_wrapper")`` and a terminal ``writer`` node, v5 is ONE
``create_agent`` supervisor whose tools are ``@tool``-wrapped subagents (the
idiom from the LangChain *subagents* doc). Each subagent is the SAME
``build_agent(AgentConfig)`` leaf v4 compiles.

Two variants are provided so the writer question can be settled empirically:

  * ``ShoppingAgentV5(variant="speaking")`` — no writer; the supervisor composes
    the final reply itself.
  * ``ShoppingAgentV5(variant="router")``   — the supervisor only routes and a
    dedicated writer composes the reply (v4's design).

``.env`` is loaded on import (same as v4) so the CLI / eval work without manual
exports.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

from agent_v5.agent import ShoppingAgentV5, TurnResult  # noqa: E402
from agent_v5.supervisor import build_supervisor  # noqa: E402

__all__ = ["ShoppingAgentV5", "TurnResult", "build_supervisor"]
