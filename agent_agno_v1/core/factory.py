"""Compile an :class:`AgentConfig` into an Agno ``Agent`` (member) or ``Team`` (leader).

The whole config → Agno mapping lives here:

  * model            → ``resolve_model`` (OpenAIChat with per-instance temperature)
  * system_prompt    → ``description`` (prepended to the system message)
  * instructions     → ``instructions`` (bullet rules in the system message)
  * tools            → registry lookups + compiled HTTP tools
  * tool_call_limit  → native ``tool_call_limit``
  * output_format    → ``output_schema`` (raw JSON-schema dict)
  * id / role        → Agno member addressing (the router delegates by ``id``)

Runtime concerns (``dependencies`` carrying the live cart, the ``db``, the
per-turn ``session_state`` snapshot) are passed in here, never baked into the
serializable config. ``add_session_state_to_context=True`` everywhere so the cart
snapshot + checkout anchor reach every model call.
"""

from __future__ import annotations

from typing import Any, Mapping

from agno.agent import Agent
from agno.team import Team, TeamMode

from agent_agno_v1.core.config import AgentConfig, to_config
from agent_agno_v1.core.models import resolve_model
from agent_agno_v1.core.tools import resolve_tools

# How many prior runs the leader keeps in context (multi-turn coherence).
_LEADER_HISTORY_RUNS = 10


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower()).strip("_") or "agent"


def build_agent(
    config: AgentConfig | Mapping[str, Any],
    *,
    dependencies: dict[str, Any] | None = None,
    db: Any | None = None,
) -> Agent:
    """Compile a declarative config into an Agno member ``Agent``."""
    cfg = to_config(config)
    return Agent(
        id=cfg.id or _slug(cfg.name),
        name=cfg.name,
        role=cfg.role or None,
        model=resolve_model(cfg.model),
        description=cfg.system_prompt,
        instructions=cfg.instructions or None,
        tools=resolve_tools(cfg.tools) or None,
        tool_call_limit=cfg.tool_call_limit,
        output_schema=cfg.output_format,
        dependencies=dependencies,
        add_session_state_to_context=True,
        db=db,
        markdown=False,
        telemetry=False,
    )


def build_team(
    config: AgentConfig | Mapping[str, Any],
    members: list[Any],
    *,
    dependencies: dict[str, Any] | None = None,
    db: Any,
    session_state: dict[str, Any] | None = None,
) -> Team:
    """Compile the coordinate-mode leader (the speaking supervisor) over its members.

    ``mode=coordinate`` makes the leader route to members AND author the final,
    user-facing reply itself — which is what streams to the client token-by-token
    (``TeamRunContent``). A ``db`` is required (history-in-context needs it).
    """
    cfg = to_config(config)
    return Team(
        members,
        id=cfg.id or _slug(cfg.name),
        name=cfg.name,
        mode=TeamMode.coordinate,
        model=resolve_model(cfg.model),
        description=cfg.system_prompt,
        instructions=cfg.instructions or None,
        tool_call_limit=cfg.tool_call_limit,
        dependencies=dependencies,
        session_state=session_state or {},
        add_session_state_to_context=True,
        overwrite_db_session_state=True,  # per-turn snapshot replaces stale state
        add_history_to_context=True,
        num_history_runs=_LEADER_HISTORY_RUNS,
        stream_member_events=True,
        store_member_responses=True,
        db=db,
        markdown=False,
        telemetry=False,
        show_members_responses=False,
    )


__all__ = ["build_agent", "build_team"]
