"""The orchestrator — pre-defined agent #1 (the router).

Its abstractions are split into siblings: :mod:`prompt` (how it routes) and
:mod:`routing` (how it resolves references across turns). It declares NO leaf
tools of its own — its "tools" are the sub-agents in
:data:`lg_agent.shopping.agents.subagents.SUBAGENTS`, wired in as delegates here.
"""

from __future__ import annotations

from typing import Any

from lg_agent.core.builder import build_agent
from lg_agent.shopping.agents.orchestrator.prompt import ROUTER_PROMPT
from lg_agent.shopping.agents.subagents import build_delegates
from lg_agent.shopping.context import ShoppingContext

MODEL = "openai:gpt-5.4-mini"

CONFIG = {
    "name": "orchestrator",
    "description": "Route the user's message to its sub-agents, then emit DONE.",
    "system_prompt": ROUTER_PROMPT,
    "model": {"provider_model": MODEL, "temperature": 0.0},
    # No `tools`: the delegates (sub-agents) are passed to build_agent() below, not
    # declared in config — they carry Python hooks and aren't registry tools.
    "middleware": [
        {"name": "empty_cart_guard", "params": {}},
        {"name": "tool_call_limit", "params": {"run_limit": 4, "exit_behavior": "end"}},
        # log_tool_calls makes each sub-agent call visible as a router/agent event.
        {"name": "log_tool_calls", "params": {"log_prefix": "orchestrator"}},
    ],
}


def build_orchestrator(store: Any | None = None) -> Any:
    """Compile the router, with the sub-agents wired in as delegate tools."""
    return build_agent(
        CONFIG,
        context_schema=ShoppingContext,
        store=store,
        delegates=build_delegates(store),
    )


__all__ = ["CONFIG", "ROUTER_PROMPT", "build_orchestrator"]
