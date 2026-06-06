"""agent_v3 — Agno port of the agent_v2 shopping assistant.

Same product (multi-agent shopping/checkout assistant), rebuilt on
Agno 2.6.x instead of LangChain/LangGraph:

  - outer LangGraph ``StateGraph``      → Agno ``Workflow`` (Loop + Router + steps)
  - ``create_agent`` sub-agents          → Agno ``Agent``
  - custom skills middleware             → native Agno ``Skills`` + a tool-gating hook
  - Pydantic ``AgentState``              → Agno ``session_state`` (a shared dict)
  - ``RuntimeContext`` (context=)        → Agno ``dependencies`` + ``run_context``
  - LangSmith (unused in v2)             → Agno tracing / AgentOS

Public surface lives in :mod:`agent_v3.workflow` (``run_turn``,
``stream_turn``, ``build_workflow``, ``fresh_state``). The package
``__init__`` stays light (just ``.env`` loading) so importing the pure
domain (``agent_v3.checkout``) doesn't construct any agents.

On import we auto-load ``.env`` from the project root AND the caller's
CWD so ``python -m agent_v3.cli`` and the server both work without a
manual ``export``.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# 1) project-root .env (sibling of the agent_v3 package dir)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
# 2) caller's CWD .env — does NOT override step 1 or the real environment.
load_dotenv()
