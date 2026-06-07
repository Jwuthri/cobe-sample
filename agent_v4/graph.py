"""Outer LangGraph wiring — GENERATED from the LEAVES registry.

  START → supervisor ─┬→ <leaf>_wrapper ─┐   (one branch per LeafSpec)
                      │      …            ┼→ supervisor (loop, max_iters)
                      └→ …               ┘
                          ↓ when supervisor decides "done"
                       writer (composes the ONE user-facing reply)
                          ↓
                       validator (retry once if the writer produced no text)
                          ↓
                       emit (append AIMessage)
                          ↓
                          END

The "1 orchestrator → n leaves → orchestrator → writer" shape is identical
to v2, but the leaf branches are no longer hand-written: ``build_graph``
iterates :data:`agent_v4.leaves.LEAVES`, compiles each declarative
``AgentConfig`` with :func:`agent_v4.configurable.build_agent`, and wires
its wrapper node. Adding a leaf = appending a ``LeafSpec`` in ``leaves.py``.
"""

from __future__ import annotations

import uuid
from typing import Any

from agent_v4.configurable import build_agent
from agent_v4.leaves import LEAVES
from agent_v4.memory import build_store
from agent_v4.registry_defaults import register_platform_defaults
from agent_v4.runtime import RuntimeContext
from agent_v4.state import MAX_VALIDATOR_RETRIES, AgentState
from agent_v4.supervisor import supervisor
from agent_v4.writer import writer
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

# ----- platform registration + module singletons -----
register_platform_defaults()
_STORE = build_store()
_CHECKPOINTER = InMemorySaver()


def _compile_leaf_agents() -> dict[str, Any]:
    """Compile every declarative leaf config into a runnable create_agent."""
    agents: dict[str, Any] = {}
    for spec in LEAVES:
        agents[spec.name] = build_agent(
            spec.config,
            checkpointer=_CHECKPOINTER if spec.needs_checkpointer else None,
            store=_STORE if spec.needs_store else None,
            context_schema=RuntimeContext,
        )
    return agents


_LEAF_AGENTS = _compile_leaf_agents()


# ============================================================ validator
def validator(state: AgentState) -> Command:
    """Minimal structural safety net: retry once if the writer produced no
    text, otherwise emit.

    All regex / keyword content checks (placeholder leaks, an unsafe-word
    blocklist, length, and the old confirmation "gate") were removed: the
    writer is an LLM that already receives the full cart state, so it owns
    content — including never claiming the order is placed unless
    ``cart.confirmed`` is true (enforced in the writer's system prompt).
    """
    if (state.draft_response or "").strip():
        return Command(goto="emit", update={"validation_errors": []})

    # Empty draft → give the writer one more try, then a graceful fallback.
    if state.response_attempts >= MAX_VALIDATOR_RETRIES:
        return Command(
            goto="emit",
            update={
                "draft_response": "Sorry, I couldn't produce a response. Could you rephrase?",
                "draft_blocks": [],
            },
        )
    return Command(
        goto="writer",
        update={
            "draft_response": None,
            "draft_blocks": [],
            "response_attempts": state.response_attempts + 1,
        },
    )


# ============================================================ emit
def emit(state: AgentState) -> Command:
    if not state.draft_response:
        return Command(goto=END, update={"done": True})
    # Carry the typed blocks alongside the message so they persist in
    # state.messages (and reach the UI via serialize_state / the bot event).
    msg = AIMessage(
        content=state.draft_response,
        additional_kwargs={"blocks": state.draft_blocks} if state.draft_blocks else {},
    )
    return Command(
        goto=END,
        update={
            "messages": [msg],
            "draft_response": None,
            "draft_blocks": [],
            "response_attempts": 0,
            "done": True,
            # Don't carry step_results / iteration across turns.
            "step_results": [],
            "iteration": 0,
        },
    )


# ============================================================ build
def build_graph():
    sg: StateGraph = StateGraph(AgentState)
    sg.add_node("supervisor", supervisor)
    # One wrapper node per leaf — generated from the registry.
    for spec in LEAVES:
        sg.add_node(f"{spec.name}_wrapper", spec.wrapper_factory(_LEAF_AGENTS[spec.name]))
    sg.add_node("writer", writer)
    sg.add_node("validator", validator)
    sg.add_node("emit", emit)
    sg.add_edge(START, "supervisor")
    return sg.compile(name="agent_v4")


graph = build_graph()


def fresh_state(user_id: str = "demo", session_id: str | None = None) -> AgentState:
    return AgentState(user_id=user_id, session_id=session_id or f"sess-{uuid.uuid4().hex[:8]}")


def run_turn(state: AgentState, user_msg: str) -> AgentState:
    state = state.model_copy(
        update={
            "messages": state.messages + [HumanMessage(content=user_msg)],
            "draft_response": None,
            "draft_blocks": [],
            "validation_errors": [],
            "response_attempts": 0,
            "step_results": [],
            "iteration": 0,
            "done": False,
        }
    )
    result = graph.invoke(state)
    return AgentState.model_validate(result)


__all__ = ["graph", "build_graph", "fresh_state", "run_turn"]
