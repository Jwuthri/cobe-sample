"""Turn/session state — the Agno analogue of agent_v2's ``AgentState``.

In agent_v2 the outer-graph state was a Pydantic ``AgentState`` with
``add_messages``/custom reducers. Agno's equivalent is a plain
``session_state`` dict that is shared by reference across every workflow
step and every SOP agent run (verified: Agno passes the same dict object
through ``run_context.session_state``). Function steps mutate it in
place; the changes are visible to later steps and persist after the run.

Layout of ``session_state``::

    user_id: str
    session_id: str
    messages: list[{"role": "human"|"ai", "content": str}]
    active_sop: str | None              # SOPName value
    skills_loaded: list[str]            # checkout skills the agent has loaded
    cart: dict                          # serialized Cart snapshot (see note)
    step_results: list[dict]            # StepResult.model_dump(mode="json")
    iteration: int
    draft_response: str | None
    validation_errors: list[dict]       # ValidationError.model_dump()
    response_attempts: int
    done: bool

Note on the cart: the rich ``Cart`` carries Decimals + invariants and is
the source of truth for checkout. During a turn the *live* ``CartService``
(in ``dependencies``) is the mutation surface for tools; each SOP step
refreshes ``session_state["cart"]`` from it (mirroring how the v2 wrapper
read the cart back into ``AgentState``) so the supervisor selector — which
only sees ``session_state`` — can read the current step.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel

from agent_v3.checkout import Cart

MAX_VALIDATOR_RETRIES = 2


class ValidationError(BaseModel):
    code: Literal["empty", "too_long", "placeholder_leak", "unsafe", "gate"]
    detail: str


def fresh_state(user_id: str = "demo", session_id: str | None = None) -> dict[str, Any]:
    """Build a clean ``session_state`` dict for a new conversation."""
    return {
        "user_id": user_id,
        "session_id": session_id or f"sess-{uuid.uuid4().hex[:8]}",
        "messages": [],
        "active_sop": None,
        "skills_loaded": [],
        "cart": Cart().model_dump(mode="json"),
        "step_results": [],
        "iteration": 0,
        "draft_response": None,
        "validation_errors": [],
        "response_attempts": 0,
        "done": False,
    }


def load_cart(session_state: dict[str, Any]) -> Cart:
    """Hydrate a live ``Cart`` from the serialized snapshot in session_state."""
    return Cart.model_validate(session_state.get("cart") or {})


def save_cart(session_state: dict[str, Any], cart: Cart) -> None:
    """Write a fresh serialized snapshot back into session_state."""
    session_state["cart"] = cart.model_dump(mode="json")


def last_user_message(session_state: dict[str, Any]) -> str:
    for m in reversed(session_state.get("messages", [])):
        if m.get("role") == "human":
            return str(m.get("content", ""))
    return ""


def reset_turn(session_state: dict[str, Any]) -> None:
    """Clear the transient per-turn fields (called at the start of a turn)."""
    session_state["active_sop"] = None
    session_state["step_results"] = []
    session_state["iteration"] = 0
    session_state["draft_response"] = None
    session_state["validation_errors"] = []
    session_state["response_attempts"] = 0
    session_state["done"] = False
