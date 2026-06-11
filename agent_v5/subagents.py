"""The three leaves, rebuilt as **tools** the supervisor can call.

This is the heart of the v5 ("agent-as-tool") topology. In v4 each leaf was a
graph node reached via ``Command(goto=...)``; here each leaf is a
``create_agent`` (built by the *unchanged* :func:`agent_v4.configurable.build_agent`)
wrapped in an ``@tool`` so the single supervisor agent invokes it as part of its
normal tool-calling loop — the idiom from
https://docs.langchain.com/oss/python/langchain/multi-agent/subagents .

Each wrapper:
  1. reads the shared :class:`~agent_v5.context.SupervisorContext` off
     ``runtime.context`` (carrying the live ``cart_service``),
  2. reconstructs the conversation for the subagent (full transcript for the
     stateless leaves; just the new instruction for checkout, which keeps its own
     checkpointer keyed by ``session_id`` — same as v4),
  3. runs the subagent with ``context=ctx`` so its tools mutate the same cart,
  4. distills a typed :class:`~agent_v4.step_result.StepResult` (reusing v4's
     extraction helpers verbatim) and appends it to ``ctx.step_results`` for the
     deterministic block-builder / writer,
  5. returns a concise summary string — the only thing the supervisor LLM reads.

Domain logic (cart diffing, "asks" per checkout step, product/serviceability/order
extraction) is imported from :mod:`agent_v4.leaves` rather than re-implemented, so
v5 renders identical blocks to v4.
"""

from __future__ import annotations

from typing import Any

from agent_v4 import ids
from agent_v4.checkout.cart import CheckoutStep
from agent_v4.configurable import build_agent
from agent_v4.leaves import (
    ALL_CHECKOUT_SKILLS,
    CHECKOUT_CONFIG,
    LEAVES_BY_NAME,
    ORDER_STATUS_CONFIG,
    PRODUCT_REC_CONFIG,
    _SUBAGENT_HISTORY_MSGS,
    _asks_for_step,
    _extract_order_from_messages,
    _extract_products_from_messages,
    _extract_serviceability_from_messages,
    checkout_anchor,
)
from agent_v4.memory import build_store
from agent_v4.registry_defaults import register_platform_defaults
from agent_v4.step_result import StepResult
from agent_v5.context import SupervisorContext, add_message_usage
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# Platform tools/skills/middleware must be registered before any leaf compiles.
register_platform_defaults()

# Module singletons — one store (long-term memory) + one checkpointer (checkout's
# per-session thread), mirroring agent_v4/graph.py.
_STORE = build_store()
_CHECKPOINTER = InMemorySaver()


def _build_leaf(config: Any, name: str) -> Any:
    """Compile one declarative leaf config into a runnable create_agent.

    ``needs_store`` / ``needs_checkpointer`` come from the SAME LeafSpec the v4
    graph uses, so the checkout leaf keeps its checkpointer and the browse leaves
    stay stateless — no behavioural drift from v4.
    """
    spec = LEAVES_BY_NAME[name]
    return build_agent(
        config,
        checkpointer=_CHECKPOINTER if spec.needs_checkpointer else None,
        store=_STORE if spec.needs_store else None,
        context_schema=SupervisorContext,
    )


_PRODUCT_REC = _build_leaf(PRODUCT_REC_CONFIG, ids.PRODUCT_REC)
_CHECKOUT = _build_leaf(CHECKOUT_CONFIG, ids.CHECKOUT)
_ORDER_STATUS = _build_leaf(ORDER_STATUS_CONFIG, ids.ORDER_STATUS)


# =============================================================================
# History reconstruction
# =============================================================================
def _clean_transcript(messages: list[BaseMessage] | None) -> list[BaseMessage]:
    """Pull a clean USER/ASSISTANT transcript out of the supervisor's state.

    The supervisor's ``state["messages"]`` is full of tool-calling noise (its own
    empty-content AIMessages + ToolMessages). A stateless subagent only wants the
    human/assistant story, so we keep HumanMessages and AIMessages that actually
    carry text.
    """
    out: list[BaseMessage] = []
    for m in messages or []:
        if isinstance(m, HumanMessage):
            out.append(m)
        elif isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            out.append(m)
    return out


def _history_window(runtime: ToolRuntime) -> list[BaseMessage]:
    """Recent clean transcript (bounded). The supervisor already routes off the
    FULL history; a subagent only needs recent turns to resolve references, and
    a bounded window keeps its tool-loop prompts small (cheaper when the
    cross-turn prompt cache has expired between user turns)."""
    state = getattr(runtime, "state", None) or {}
    return _clean_transcript(state.get("messages"))[-_SUBAGENT_HISTORY_MSGS:]


def _subagent_messages(runtime: ToolRuntime, query: str) -> list[BaseMessage]:
    """Recent transcript + the supervisor's instruction as the final human turn."""
    return [*_history_window(runtime), HumanMessage(content=query)]


def _ctx(runtime: ToolRuntime) -> SupervisorContext:
    return runtime.context


# =============================================================================
# product_rec — browse + cart management (stateless leaf, full history)
# =============================================================================
@tool("product_rec")
def call_product_rec(query: str, runtime: ToolRuntime[SupervisorContext] = None) -> str:
    """Search the catalog, look up a product, answer delivery-area questions, and
    edit the cart (add / remove / change quantity / show contents).

    Call this for ANY browsing or cart-content request — including "add X",
    "remove the hoodie", "what's in my cart", "do you ship to 94110". Pass a
    self-contained instruction as ``query`` (e.g. "add P-2 to the cart", "search
    for caps"). Adding an item is a natural cue to proceed to ``checkout`` next.
    """
    ctx = _ctx(runtime)
    cart = ctx.cart_service.cart
    before = {i.product_id: i.quantity for i in cart.items}

    # Build: [recent history, (cart note), instruction]. The cart note is a
    # volatile SUFFIX (right before the instruction), not a prefix — a changing
    # note at the front would break the cacheable history prefix every turn.
    messages: list[BaseMessage] = list(_history_window(runtime))
    if cart.items:
        cart_note = (
            "Current cart: "
            + "; ".join(f"{i.product_id} {i.name} x{i.quantity}" for i in cart.items)
            + ". To edit it, use remove_item / set_quantity — do NOT search the "
            "catalog to remove or change an item already in the cart."
        )
        messages.append(SystemMessage(content=cart_note))
    messages.append(HumanMessage(content=query))

    result = _PRODUCT_REC.invoke({"messages": messages}, context=ctx)
    out_messages = result["messages"]
    add_message_usage(ctx.subagent_usage, out_messages)

    products = _extract_products_from_messages(out_messages)
    serviceability = _extract_serviceability_from_messages(out_messages)
    viewed_cart = any(
        getattr(m, "name", None) == "get_cart_summary" for m in out_messages
    )

    after = {i.product_id: i.quantity for i in cart.items}
    added = [pid for pid in after if after[pid] > before.get(pid, 0)]
    removed = [pid for pid in before if pid not in after]
    decreased = [pid for pid in after if pid in before and after[pid] < before[pid]]
    cart_changed = bool(added or removed or decreased)

    def _cart_lines() -> list[dict]:
        return [
            {"id": i.product_id, "name": i.name, "qty": i.quantity, "price": str(i.unit_price)}
            for i in cart.items
        ]

    next_sop: str | None = None
    asks: list[str] = []
    details: dict | None = None

    if added:
        summary = f"added {', '.join(added)} to cart"
        next_sop = ids.CHECKOUT
        details = {"added": added}
        if products:
            details["products"] = products
    elif removed or decreased:
        changed = removed + decreased
        verb = "removed" if removed and not decreased else "updated"
        summary = f"{verb} cart ({', '.join(changed)})"
        details = {"cart_edit": {"removed": removed, "decreased": decreased, "items": _cart_lines()}}
    elif serviceability:
        summary = "answered a serviceability question"
        details = {"serviceability": serviceability}
        if products:
            details["products"] = products
    elif products:
        summary = f"catalog returned {len(products)} matching product(s)"
        asks = ["pick a product id (e.g. P-1) to add to your cart"]
        details = {"products": products}
    elif viewed_cart and cart.items:
        summary = "showed the cart"
        details = {"cart_edit": {"removed": [], "decreased": [], "items": _cart_lines()}}
    else:
        summary = "no products matched the user's query"
        asks = ["clarify what you're looking for"]

    ctx.step_results.append(
        StepResult(
            sop=ids.PRODUCT_REC,
            summary=summary,
            asks=asks,
            next_sop=next_sop,
            details=details,
            cart_diff={"items": len(cart.items)} if cart_changed else None,
        )
    )
    hint = " (you can proceed to checkout)" if next_sop == ids.CHECKOUT else ""
    return f"{summary}{hint}"


# =============================================================================
# checkout — identity → payment (stateful leaf, keeps its own checkpointer)
# =============================================================================
@tool("checkout")
def call_checkout(query: str, runtime: ToolRuntime[SupervisorContext] = None) -> str:
    """Move an order forward: capture identity, shipping address, delivery option,
    and payment, then place the order ONLY on the user's explicit "yes".

    Requires items already in the cart (use ``product_rec`` to add them first).
    Pass the user's latest checkout-relevant message as ``query`` (their name, an
    address, a delivery choice, a payment method, or "yes" to confirm).
    """
    ctx = _ctx(runtime)
    # Stateless + cart-anchored: the shared cart is the source of truth, so we
    # inject an authoritative progress block and pre-unlock every skill, then run
    # checkout fresh. No checkpointed thread to re-walk → it does only the next
    # step (the old design re-executed the whole flow every turn). Because the run
    # is fresh, ``result["messages"]`` is just this turn's, so usage is accurate.
    result = _CHECKOUT.invoke(
        {
            "messages": [
                SystemMessage(content=checkout_anchor(ctx.cart_service.cart)),
                HumanMessage(content=query),
            ],
            "skills_loaded": list(ALL_CHECKOUT_SKILLS),
        },
        context=ctx,
    )
    add_message_usage(ctx.subagent_usage, result["messages"])

    cart = ctx.cart_service.cart
    asks: list[str] = []
    if cart.step.value.startswith("collecting_"):
        asks = _asks_for_step(cart.step.value, cart)
    elif cart.ready_to_confirm() and not cart.confirmed:
        asks = ["explicit yes to place the order"]

    summary = f"checkout subagent finished at step={cart.step.value}; items={len(cart.items)}"
    ctx.step_results.append(
        StepResult(
            sop=ids.CHECKOUT,
            summary=summary,
            asks=asks,
            next_sop=None,
            cart_diff={"step": cart.step.value},
        )
    )
    asks_note = f" Needs from user: {', '.join(asks)}." if asks else ""
    confirmed = " ORDER CONFIRMED." if cart.confirmed else ""
    return f"{summary}.{asks_note}{confirmed}"


# =============================================================================
# order_status — past-order lookup (stateless leaf, full history)
# =============================================================================
@tool("order_status")
def call_order_status(query: str, runtime: ToolRuntime[SupervisorContext] = None) -> str:
    """Look up a PAST order's status / tracking (order ids look like ORD-* or
    RCPT-*). Pass the user's order question as ``query``."""
    ctx = _ctx(runtime)
    result = _ORDER_STATUS.invoke({"messages": _subagent_messages(runtime, query)}, context=ctx)
    add_message_usage(ctx.subagent_usage, result["messages"])
    order_details = _extract_order_from_messages(result["messages"])
    summary = "looked up order status" if order_details else "could not find a matching order"
    ctx.step_results.append(
        StepResult(
            sop=ids.ORDER_STATUS,
            summary=summary,
            asks=[] if order_details else ["confirm the order id"],
            next_sop=None,
            details=order_details,
        )
    )
    return summary


SUBAGENT_TOOLS = [call_product_rec, call_checkout, call_order_status]

# Re-exported so the empty-cart guard can reference the checkout tool by name.
CHECKOUT_TOOL_NAME = "checkout"

__all__ = [
    "SUBAGENT_TOOLS",
    "CHECKOUT_TOOL_NAME",
    "call_product_rec",
    "call_checkout",
    "call_order_status",
]
