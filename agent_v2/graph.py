"""Outer LangGraph wiring (v4 loop architecture).

  START → supervisor ─┬→ checkout_wrapper ─┐
                      ├→ product_rec_wrapper ┼→ supervisor (loop, max_iters)
                      └→ order_status_wrapper ┘
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

Wrappers return ``StepResult`` records (accumulated in
``state.step_results``) instead of raw text. The writer is the single
voice that talks to the user.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from agent_v2 import debug_log
from agent_v2.checkout import CartService
from agent_v2.memory import build_store
from agent_v2.runtime import RuntimeContext
from agent_v2.sops import (
    build_checkout_agent,
    build_order_status_agent,
    build_product_rec_agent,
)
from agent_v2.state import MAX_VALIDATOR_RETRIES, AgentState, ValidationError
from agent_v2.step_result import StepResult
from agent_v2.supervisor import SOPName, supervisor
from agent_v2.writer import writer
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Command

# ----- module singletons -----
_STORE: BaseStore = build_store()
_CHECKPOINTER = InMemorySaver()
_CHECKOUT_AGENT = build_checkout_agent(checkpointer=_CHECKPOINTER, store=_STORE)
_ORDER_STATUS_AGENT = build_order_status_agent(store=_STORE)
_PRODUCT_REC_AGENT = build_product_rec_agent(store=_STORE)


def _runtime_context(state: AgentState) -> RuntimeContext:
    return RuntimeContext(
        user_id=state.user_id,
        session_id=state.session_id,
        cart_service=CartService(state.cart),
    )


def _stream_subagent(
    agent: Any,
    input_state: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    context: RuntimeContext | None = None,
) -> dict[str, Any]:
    """Run a subagent via ``stream`` so ``log_tool_calls`` custom events reach the outer SSE/CLI.

    ``invoke()`` does not propagate custom stream chunks to the parent graph; streaming
    and re-emitting via ``get_stream_writer()`` fixes missing SKILL/TOOL lines in the UI.
    """
    writer = None
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None

    result: dict[str, Any] | None = None
    for chunk in agent.stream(
        input_state,
        config=config,
        context=context,
        stream_mode=["custom", "values"],
    ):
        if isinstance(chunk, tuple) and len(chunk) == 2:
            mode, payload = chunk
        else:
            mode, payload = "values", chunk
        if mode == "custom" and writer is not None and isinstance(payload, dict):
            writer(payload)
        elif mode == "values" and isinstance(payload, dict):
            result = payload

    if result is not None:
        return result
    return agent.invoke(input_state, config=config, context=context)


def _last_ai_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            return str(m.content)
    return ""


# Matches lines produced by catalog_tools.search_products / get_product:
#   "P-2: Black Hoodie — $49.99 [apparel, hoodie, black]"
_PRODUCT_LINE_RE = re.compile(r"^(P-\d+):\s+(.+?)\s+[—\-]\s+\$(\S+)\s+\[(.+?)\]$")


def _extract_products_from_messages(messages) -> list[dict]:
    """Walk ToolMessages from search_products / get_product and parse them
    into a structured list the writer can render. De-dups by product id."""
    products: list[dict] = []
    seen: set[str] = set()
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) not in ("search_products", "get_product"):
            continue
        for line in str(m.content).splitlines():
            match = _PRODUCT_LINE_RE.match(line.strip())
            if not match:
                continue
            pid = match.group(1)
            if pid in seen:
                continue
            seen.add(pid)
            products.append(
                {
                    "id": pid,
                    "name": match.group(2),
                    "price": match.group(3),
                    "tags": [t.strip() for t in match.group(4).split(",")],
                }
            )
    return products


def _extract_serviceability_from_messages(messages) -> dict | None:
    """Pull the most recent check_serviceability tool result, if any."""
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) != "check_serviceability":
            continue
        content = str(m.content).strip()
        if not content:
            continue
        return {"raw": content}
    return None


def _extract_order_from_messages(messages) -> dict | None:
    """Pull the raw order-status text out of the subagent's tool result."""
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) not in ("get_order_status", "list_recent_orders"):
            continue
        content = str(m.content).strip()
        if content and "unknown order" not in content.lower():
            return {"raw": content}
    return None


# ============================================================ wrappers
def checkout_wrapper(state: AgentState) -> Command:
    """Run the checkout subagent for one iteration; return a StepResult."""
    ctx = _runtime_context(state)
    cfg = {"configurable": {"thread_id": state.session_id}}
    debug_log.graph(
        "checkout_wrapper",
        f"start step={state.cart.step.value} skills={state.skills_loaded} "
        f"msg={state.last_user_message()[:100]!r}",
    )
    result = _stream_subagent(
        _CHECKOUT_AGENT,
        {
            "messages": [HumanMessage(content=state.last_user_message())],
            "skills_loaded": list(state.skills_loaded),
        },
        config=cfg,
        context=ctx,
    )
    cart = ctx.cart_service.cart
    debug_log.graph(
        "checkout_wrapper",
        f"done step={cart.step.value} skills={result.get('skills_loaded', [])} items={len(cart.items)}",
    )

    asks: list[str] = []
    if cart.step.value.startswith("collecting_"):
        # Map the current step to a plain-english ask the writer can echo.
        asks = _asks_for_step(cart.step.value, cart)
    elif cart.ready_to_confirm() and not cart.confirmed:
        # Prompt-gated confirmation: signal to the writer that it should
        # present the summary and ask for explicit approval.
        asks = ["explicit yes to place the order"]

    step_summary = (
        f"checkout subagent finished at step={cart.step.value}; " f"items={len(cart.items)}"
    )

    sr = StepResult(
        sop=SOPName.CHECKOUT,
        summary=step_summary,
        asks=asks,
        next_sop=None,  # supervisor decides whether to keep going
        cart_diff={"step": cart.step.value},
    )
    return Command(
        goto="supervisor",
        update={
            "cart": cart,
            "skills_loaded": result.get("skills_loaded", []),
            "step_results": [sr],
        },
    )


_PRODUCT_REC_HISTORY_TURNS = 8


def product_rec_wrapper(state: AgentState) -> Command:
    """Run the product_rec subagent for one iteration.

    Pre-conditions:
      - The subagent receives the last N turns of conversation so it
        can resolve pronouns like "them" / "those" / "the sneakers".
    Post-conditions:
      - We observe whether the subagent CALLED add_item (the cart's
        item count grew) and signal next_sop=checkout if so. This
        replaces the old regex-based ``_picked_product_id`` heuristic.
    """
    ctx = _runtime_context(state)
    items_before = len(state.cart.items)

    # Pass recent conversation history. We use the model's own messages
    # — if the user just saw "P-3 Running Sneakers" in the assistant's
    # prior turn, the subagent now sees that too and can resolve "them"
    # to "P-3" without us regex-matching.
    history = state.messages[-_PRODUCT_REC_HISTORY_TURNS:]
    if not history:
        history = [HumanMessage(content=state.last_user_message())]

    result = _stream_subagent(_PRODUCT_REC_AGENT, {"messages": history}, context=ctx)

    products = _extract_products_from_messages(result["messages"])
    serviceability = _extract_serviceability_from_messages(result["messages"])

    cart_now = ctx.cart_service.cart
    added_ids = [i.product_id for i in cart_now.items[items_before:]]  # items appended this turn

    next_sop = None
    asks: list[str] = []
    details: dict | None = None

    if added_ids:
        summary = f"added {', '.join(added_ids)} to cart"
        next_sop = SOPName.CHECKOUT
        details = {"added": added_ids}
        if products:
            details["products"] = products
    elif serviceability:
        summary = "answered a serviceability question"
        details = {"serviceability": serviceability}
        if products:
            details["products"] = products
    elif products:
        summary = f"catalog returned {len(products)} matching product(s)"
        asks = ["pick a product id (e.g. P-1) to add to your cart"]
        details = {"products": products}
    else:
        # Nothing matched. The writer should ask the user to clarify.
        summary = "no products matched the user's query"
        asks = ["clarify what you're looking for"]

    sr = StepResult(
        sop=SOPName.PRODUCT_REC,
        summary=summary,
        asks=asks,
        next_sop=next_sop,
        details=details,
        cart_diff={"items": len(cart_now.items)} if added_ids else None,
    )
    return Command(
        goto="supervisor",
        update={"cart": cart_now, "step_results": [sr]},
    )


def order_status_wrapper(state: AgentState) -> Command:
    ctx = _runtime_context(state)
    result = _stream_subagent(
        _ORDER_STATUS_AGENT,
        {"messages": [HumanMessage(content=state.last_user_message())]},
        context=ctx,
    )
    order_details = _extract_order_from_messages(result["messages"])
    sr = StepResult(
        sop=SOPName.ORDER_STATUS,
        summary=("looked up order status" if order_details else "could not find a matching order"),
        asks=[] if order_details else ["confirm the order id"],
        next_sop=None,
        details=order_details,
    )
    return Command(goto="supervisor", update={"step_results": [sr]})


def _asks_for_step(step_value: str, cart) -> list[str]:
    if step_value == "collecting_identity":
        return ["first name", "last name"]
    if step_value == "collecting_address":
        return ["street", "city", "state", "zip code"]
    if step_value == "awaiting_serviceability":
        return ["(internal: serviceability lookup)"]
    if step_value == "collecting_delivery":
        opts = ", ".join(cart.serviceable_options) or "available delivery options"
        return [f"delivery option ({opts})"]
    if step_value == "collecting_payment":
        return ["payment method (card / cash / wallet)", "card_token if paying by card"]
    return []


# ============================================================ gate
def checkout_gate(state: AgentState) -> Command:
    """Re-assert ``Cart.blockers()`` if the writer's reply claims confirmed."""
    if state.active_sop != SOPName.CHECKOUT:
        return Command(goto="validator")
    cart = state.cart
    text_lower = (state.draft_response or "").lower()
    claims_done = any(
        k in text_lower for k in ("confirmed", "placed", "your order", "all set", "thank you")
    )
    if claims_done and not cart.ready_to_confirm():
        blockers = "; ".join(b.code for b in cart.blockers())
        return Command(
            goto="supervisor",
            update={
                "draft_response": None,
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
                "validation_errors": errors,
            },
        )
    # Retry: bounce back to writer (cheap to re-run).
    return Command(
        goto="writer",
        update={
            "validation_errors": errors,
            "draft_response": None,
            "response_attempts": state.response_attempts + 1,
        },
    )


# ============================================================ emit
def emit(state: AgentState) -> Command:
    if not state.draft_response:
        return Command(goto=END, update={"done": True})
    msg = AIMessage(content=state.draft_response)
    return Command(
        goto=END,
        update={
            "messages": [msg],
            "draft_response": None,
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
    sg.add_node("checkout_wrapper", checkout_wrapper)
    sg.add_node("product_rec_wrapper", product_rec_wrapper)
    sg.add_node("order_status_wrapper", order_status_wrapper)
    sg.add_node("writer", writer)
    sg.add_node("checkout_gate", checkout_gate)
    sg.add_node("validator", validator)
    sg.add_node("emit", emit)
    sg.add_edge(START, "supervisor")
    return sg.compile(name="agent_v2")


graph = build_graph()


def fresh_state(user_id: str = "demo", session_id: str | None = None) -> AgentState:
    return AgentState(user_id=user_id, session_id=session_id or f"sess-{uuid.uuid4().hex[:8]}")


def run_turn(state: AgentState, user_msg: str) -> AgentState:
    state = state.model_copy(
        update={
            "messages": state.messages + [HumanMessage(content=user_msg)],
            "draft_response": None,
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
