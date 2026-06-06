"""Outer LangGraph wiring — GENERATED from the LEAVES registry.

  START → supervisor ─┬→ <leaf>_wrapper ─┐   (one branch per LeafSpec)
                      │      …            ┼→ supervisor (loop, max_iters)
                      └→ …               ┘
                          ↓ when supervisor decides "done"
                       writer (composes the ONE user-facing reply)
                          ↓
                       checkout_gate (re-asserts Cart.blockers if cart claims confirmed)
                          ↓
                       validator (format/safety/retry)
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

import re
import uuid
from typing import Any

from agent_v4 import ids
from agent_v4.configurable import build_agent
from agent_v4.leaves import LEAVES
from agent_v4.memory import build_store
from agent_v4.registry_defaults import register_platform_defaults
from agent_v4.runtime import RuntimeContext
from agent_v4.state import MAX_VALIDATOR_RETRIES, AgentState, ValidationError
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


# ============================================================ gate
# Phrases that ASSERT the order is complete/placed. Kept specific so the gate
# doesn't false-trigger on the writer merely describing the cart (e.g. "your
# order has 2 hoodies"), being polite ("thank you"), or prompting confirmation
# ("reply yes to place the order").
_CONFIRM_CLAIM_PHRASES = (
    "order is confirmed",
    "order confirmed",
    "order has been placed",
    "order is placed",
    "order placed",
    "placed your order",
    "successfully placed",
    "successfully ordered",
    "your order is on its way",
    "on its way",
    "you're all set",
    "you are all set",
    "here is your receipt",
    "here's your receipt",
    "receipt id",
)


def checkout_gate(state: AgentState) -> Command:
    """Re-assert ``Cart.blockers()`` if the writer's reply claims confirmed."""
    if state.active_sop != ids.CHECKOUT:
        return Command(goto="validator")
    cart = state.cart
    text_lower = (state.draft_response or "").lower()
    claims_done = any(p in text_lower for p in _CONFIRM_CLAIM_PHRASES)
    if claims_done and not cart.ready_to_confirm():
        blockers = "; ".join(b.code for b in cart.blockers())
        return Command(
            goto="supervisor",
            update={
                "draft_response": None,
                "draft_blocks": [],
                "validation_errors": [
                    ValidationError(
                        code="gate",
                        detail=f"model claimed confirm but blockers remain: {blockers}",
                    )
                ],
                "response_attempts": state.response_attempts + 1,
                # Clear step_results so the next loop starts fresh.
                "step_results": [],
            },
        )
    return Command(goto="validator")


# ============================================================ validator
_PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}|<[A-Z_]+>")
_UNSAFE_RE = re.compile(r"\b(damn|hate you|stupid customer)\b", re.I)
MAX_RESPONSE_CHARS = 2000


def validator(state: AgentState) -> Command:
    draft = (state.draft_response or "").strip()
    errors: list[ValidationError] = []
    if not draft:
        errors.append(ValidationError(code="empty", detail="writer produced no text"))
    else:
        if len(draft) > MAX_RESPONSE_CHARS:
            errors.append(ValidationError(code="too_long", detail=f"{len(draft)} chars"))
        if _PLACEHOLDER_RE.search(draft):
            errors.append(
                ValidationError(code="placeholder_leak", detail="unfilled template token")
            )
        if _UNSAFE_RE.search(draft):
            errors.append(ValidationError(code="unsafe", detail="safety blocklist hit"))

    if not errors:
        return Command(goto="emit", update={"validation_errors": []})

    if state.response_attempts >= MAX_VALIDATOR_RETRIES:
        return Command(
            goto="emit",
            update={
                "draft_response": "Sorry, I couldn't produce a clean response. Could you rephrase?",
                "draft_blocks": [],
                "validation_errors": errors,
            },
        )
    # Retry: bounce back to writer (cheap to re-run).
    return Command(
        goto="writer",
        update={
            "validation_errors": errors,
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
    sg.add_node("checkout_gate", checkout_gate)
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
