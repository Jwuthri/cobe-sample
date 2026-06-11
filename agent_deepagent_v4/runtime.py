"""Runtime harness: build the agent once, drive turns, persist carts.

Two parallel persistences keep a session coherent:
  * **Messages** live in the deepagents checkpointer, keyed by ``thread_id``
    (= ``session_id``). We pass only the NEW user message each turn.
  * **The cart** lives in ``SESSIONS`` (one ``CartService`` per session) and is
    handed to every turn via ``ShopContext`` so all subagents mutate one cart.

``run_turn`` returns a :class:`TurnResult`. When the safe-checkout flow pauses
for approval, ``result.interrupt`` is populated and ``result.reply`` is ``None``;
call :func:`resume_turn` with the human's decision to finish the turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent_deepagent_v4.agents.orchestrator.agent import build_orchestrator
from agent_deepagent_v4.config import load_env
from agent_deepagent_v4.context import ShopContext
from agent_deepagent_v4.domain.cart import Cart
from agent_deepagent_v4.domain.memory import build_store
from agent_deepagent_v4.domain.service import CartService
from agent_deepagent_v4.messages import text_of

# Module singletons — one store + checkpointer + compiled agent for the process.
_STORE = build_store()
_CHECKPOINTER = InMemorySaver()
_AGENT: Any | None = None

# session_id -> the live CartService for that conversation.
SESSIONS: dict[str, CartService] = {}


def get_agent() -> Any:
    """Lazily build (and cache) the orchestrator deep agent."""
    global _AGENT
    if _AGENT is None:
        load_env()
        _AGENT = build_orchestrator(checkpointer=_CHECKPOINTER, store=_STORE)
    return _AGENT


def cart_service_for(session_id: str) -> CartService:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = CartService(Cart())
    return SESSIONS[session_id]


def reset_session(session_id: str) -> None:
    SESSIONS.pop(session_id, None)


def _d(value: Any) -> Any:
    return str(value) if isinstance(value, Decimal) else value


def cart_snapshot(cart: Cart) -> dict[str, Any]:
    """JSON-able snapshot of the cart for UIs / CLI / tests."""
    return {
        "step": cart.step.value,
        "cart_id": cart.cart_id,
        "items": [
            {"id": i.product_id, "name": i.name, "qty": i.quantity, "unit_price": _d(i.unit_price)}
            for i in cart.items
        ],
        "customer": cart.customer.model_dump(),
        "address": cart.address.model_dump(),
        "serviceable": cart.serviceable,
        "serviceable_options": list(cart.serviceable_options),
        "delivery_option": cart.delivery_option,
        "payment_method": cart.payment_method,
        "subtotal": _d(cart.subtotal),
        "grand_total": _d(cart.grand_total) if cart.grand_total is not None else None,
        "blockers": [{"code": b.code, "message": b.message} for b in cart.blockers()],
        "ready_to_confirm": cart.ready_to_confirm(),
        "confirmed": cart.confirmed,
        "receipt_id": cart.receipt_id,
    }


@dataclass
class TurnResult:
    reply: str | None
    cart: dict[str, Any]
    interrupt: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_approval(self) -> bool:
        return self.interrupt is not None


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    intr = result.get("__interrupt__")
    if not intr:
        return None
    first = intr[0]
    return getattr(first, "value", first)


def _finalize(result: dict[str, Any], session_id: str) -> TurnResult:
    cart = cart_service_for(session_id).cart
    interrupt = _interrupt_payload(result)
    if interrupt is not None:
        return TurnResult(reply=None, cart=cart_snapshot(cart), interrupt=interrupt, raw=result)
    messages = result.get("messages") or []
    reply = text_of(messages[-1]).strip() if messages else ""
    return TurnResult(reply=reply or None, cart=cart_snapshot(cart), raw=result)


def run_turn(
    session_id: str, message: str, *, user_id: str = "demo", require_approval: bool = True
) -> TurnResult:
    """Run one conversational turn. May return an approval-pending result."""
    agent = get_agent()
    ctx = ShopContext(
        user_id=user_id,
        session_id=session_id,
        cart_service=cart_service_for(session_id),
        require_approval=require_approval,
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        context=ctx,
        config={"configurable": {"thread_id": session_id}},
    )
    return _finalize(result, session_id)


def resume_turn(
    session_id: str, decision: dict[str, Any], *, user_id: str = "demo", require_approval: bool = True
) -> TurnResult:
    """Resume a turn paused at the checkout approval step.

    ``decision`` is e.g. ``{"approved": True}`` or ``{"approved": False, "reason": "..."}``.
    """
    agent = get_agent()
    ctx = ShopContext(
        user_id=user_id,
        session_id=session_id,
        cart_service=cart_service_for(session_id),
        require_approval=require_approval,
    )
    result = agent.invoke(
        Command(resume=decision),
        context=ctx,
        config={"configurable": {"thread_id": session_id}},
    )
    return _finalize(result, session_id)
