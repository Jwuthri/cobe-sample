"""agent_v4 — declarative, config-driven leaves on the v2 LangGraph shape.

Same runtime architecture as agent_v2 (1 orchestrator → n leaves →
orchestrator → writer), but each leaf is defined by a JSON-serializable
:class:`~agent_v4.configurable.AgentConfig` and compiled by
:func:`~agent_v4.configurable.build_agent`, and the graph topology is
generated from the :data:`~agent_v4.leaves.LEAVES` registry.

Public surface:
  - ``build_graph()`` / ``graph``        — compile / singleton outer graph
  - ``AgentState`` / ``fresh_state`` / ``run_turn``
  - ``AgentConfig`` / ``build_agent``    — the declarative leaf builder
  - ``build_catalog`` / ``register_platform_defaults`` — builder-UI surface
  - ``LEAVES``                           — the leaf registry

On import we auto-load ``.env`` from the package dir AND the user's CWD so
``python -m agent_v4.cli`` and ``langgraph dev`` both work without manual
``export`` commands.
"""

from pathlib import Path

from dotenv import load_dotenv

# 1) package-local .env (when running from the repo root)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
# 2) caller's CWD .env — does NOT override values already set above.
load_dotenv()

from agent_v4.configurable import (  # noqa: E402
    AgentConfig,
    build_agent,
    build_catalog,
)
from agent_v4.graph import build_graph, fresh_state, graph, run_turn  # noqa: E402
from agent_v4.leaves import LEAVES  # noqa: E402
from agent_v4.registry_defaults import register_platform_defaults  # noqa: E402
from agent_v4.state import AgentState  # noqa: E402

__all__ = [
    "build_graph",
    "graph",
    "AgentState",
    "fresh_state",
    "run_turn",
    "AgentConfig",
    "build_agent",
    "build_catalog",
    "register_platform_defaults",
    "LEAVES",
]
