"""agent_v2 — LangChain v1 + LangGraph hybrid agent pipeline.

Public surface:
  - ``build_graph()``       — compile a fresh graph instance
  - ``graph``               — module-level singleton (used by Studio)
  - ``AgentState``          — outer state schema
  - ``fresh_state``         — convenience factory
  - ``run_turn``            — one-turn helper for tests/CLI

See ``README.md`` for the architecture diagram and setup steps.

On import we auto-load ``.env`` from the package dir AND the user's
CWD so ``python -m agent_v2.cli`` and ``langgraph dev`` both work
without manual ``export`` commands.
"""

from pathlib import Path

from dotenv import load_dotenv

# 1) package-local .env (when running from the agent_v2 dir)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
# 2) caller's CWD .env (when imported from another project) — does NOT
#    override values already set by step 1 or by the actual environment.
load_dotenv()

from agent_v2.graph import build_graph, fresh_state, graph, run_turn  # noqa: E402
from agent_v2.state import AgentState  # noqa: E402

__all__ = ["build_graph", "graph", "AgentState", "fresh_state", "run_turn"]
