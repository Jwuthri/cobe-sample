"""The agent leaves — declared as data, wired as a registry.

In v2 each leaf was a hand-written ``build_*_agent`` (``sops/*.py``) plus a
bespoke ``*_wrapper`` node in ``graph.py``. v4 splits each leaf into:

  1. an :class:`~agent_v4.configurable.AgentConfig` — the *declarative*
     definition (model, prompt, tools, skills, middleware). This is the
     part the design doc makes config-driven; ``build_agent`` compiles it
     into the same ``create_agent`` call v2 wrote by hand.
  2. a **wrapper** — the domain post-processing that turns the leaf's tool
     calls into a :class:`~agent_v4.step_result.StepResult` for the
     supervisor + writer. This stays Python (it's genuine domain logic),
     but is registered alongside the config.

Both halves live in one :class:`LeafSpec`, collected in :data:`LEAVES`.
The outer graph and the supervisor are *generated* from this list, so
"1 orchestrator → n leaves → orchestrator → writer" scales by appending a
spec here — no edits to ``graph.py`` or ``supervisor.py``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_v4 import debug_log, ids
from agent_v4.checkout import CartService
from agent_v4.configurable import AgentConfig, MiddlewareSpec, RegistryTool, SkillSpec
from agent_v4.runtime import RuntimeContext
from agent_v4.skills import CHECKOUT_SKILLS
from agent_v4.state import AgentState
from agent_v4.step_result import StepResult
from agent_v4.tools.catalog_tools import get_product, search_products
from agent_v4.tools.checkout_tools import (
    CHECKOUT_TOOLS,
    add_item,
    get_cart_summary,
    remove_item,
    set_quantity,
)
from agent_v4.tools.order_tools import get_order_status, list_recent_orders
from agent_v4.tools.serviceability_tools import check_serviceability
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.types import Command

# =============================================================================
# System prompts (verbatim from v2 sops/*; the *definition* is now data)
# =============================================================================
CHECKOUT_SYSTEM_PROMPT = """\
You are the checkout assistant. You guide the user through placing an order.

The checkout flow has FIVE sub-skills, loaded in order:
  1. collect_identity         (capture first/last name)
  2. collect_address          (capture shipping address)
  3. lookup_serviceability    (verify we ship there + which options)
  4. collect_delivery         (pick a delivery option, quote shipping/tax)
  5. collect_payment          (capture payment method)

Always call ``load_skill(name)`` BEFORE you call any tool that skill
unlocks. The available-skills block in the system prompt shows what's
already loaded.

You can call ``get_cart_summary()`` any time to see the current cart
state and any blockers.

Items are already in the cart from a previous step (e.g., handed off
from product recommendations). You do NOT have an add-item tool — never
try to add products. Use get_cart_summary() to inspect the cart if you
need to, then proceed to the next missing piece of the checkout flow.

## Confirmation rule (read carefully)

NEVER call ``confirm_checkout`` automatically when the cart becomes
ready_to_confirm. Instead:
  1. Call ``get_cart_summary()`` so the user can see the order.
  2. Stop — let the writer present the summary and ask the user to
     confirm.
  3. ONLY call ``confirm_checkout`` on a SUBSEQUENT turn, when the
     user's most recent message is an explicit approval like "yes",
     "y", "confirm", "place the order", "go ahead", "do it".
  4. If the user pushes back ("wait", "no", "actually change…"),
     DO NOT call confirm_checkout — handle their request instead.

You no longer speak directly to the user — the writer agent produces
the final reply. Just do your work via tool calls and stop when
you've made progress; the writer will summarize.
"""

PRODUCT_REC_PROMPT = """\
You handle browsing AND cart management: search, look up products,
answer serviceability, and edit the cart (add, remove, change quantity,
and show what's in it).

Your tools:
  - search_products(query, limit=5) — find products by free-text query.
                                       Empty query returns the full catalog.
  - get_product(product_id)         — fetch a single product by id.
  - check_serviceability(zip_code)  — does the store ship to a given
                                       zip, and with what options?
  - add_item(product_id, quantity=1) — add a product to the user's cart.
  - remove_item(product_id)          — remove a product line from the cart.
  - set_quantity(product_id, qty)    — set a line's quantity (0 removes it).
  - get_cart_summary()               — show the current cart + totals.

## Which tool to use

  * "do you sell X / what do you offer / show me the catalog / find me a Y"
    → search_products with the relevant query.

  * "tell me about P-2" → get_product("P-2").

  * "do you deliver to <city/zip>", "what shipping for 94110", "yes, <zip>"
    (after we just asked for zip) → check_serviceability. If the user
    gave a city without a zip, ask them for the zip.

  * **"add the X to my cart", "I'll take the sneakers", "buy P-3", "add
    them", "yeah those", "yes please", "the cheaper one", "the green
    one"** → add_item with the right product_id. Use the conversation
    history to resolve what the user is referring to:
      - If you JUST showed a single product, "it" / "that" / "them" /
        "the X" refers to that product.
      - If you showed multiple, "the cheaper one" / "the red one" /
        "the second one" / "the X one" refers by attribute.
      - If you can't tell which one, ASK before adding.
    Product ids are case- and hyphen-insensitive in user speech ("p3",
    "P3", "p-3", "P-3" all mean P-3) — pass the canonical "P-N" form
    to add_item.

  * **"remove the hoodie", "remove P-2", "take the cap out", "delete the
    second one", "I don't want the shoes anymore"** → this is a CART edit,
    NOT a search. Do NOT call search_products. The current cart contents
    are shown to you above (or call get_cart_summary); find the matching
    line and call remove_item(product_id) — or set_quantity(product_id, 0).

  * **"make it 2", "change the hoodie to 3", "I want two of those"** →
    set_quantity(product_id, qty). set_quantity(product_id, 0) removes it.

  * **"what's in my cart", "show my cart", "why are there 2 hoodies"** →
    get_cart_summary() and report the contents. Do NOT add or remove
    anything for a pure "what's in my cart" question.

## Rules

  - Never invent products. Only mention what the tools return.
  - Order ids (formats like ORD-123 or RCPT-9000) are NOT products.
    NEVER pass them to get_product or search_products — a separate agent
    handles order status. If the message mixes a product question with an
    order-status question (e.g. "what hoodies do you have, and where's my
    order ORD-7?"), handle ONLY the product part here and ignore the
    order id.
  - If a search returns no matches, say so and ask the user to clarify.
  - If the user already saw a product list in recent turns and now says
    "yes" / "add it" / "buy it", DON'T re-search — go straight to
    add_item.
  - Be concise. The writer composes the final user-facing reply; you
    just do the work via tool calls.
"""

ORDER_STATUS_PROMPT = """\
You help users check the status of their orders.

If the user mentions a specific order id (looks like ORD-* or RCPT-*),
call get_order_status. Otherwise call list_recent_orders to show what
they have, then ask which one they want details on.

Be concise. Report the order id, status, and tracking URL if any.
"""

# =============================================================================
# Declarative leaf configs
# =============================================================================
def _registry_tools(tools: list[Any]) -> list[RegistryTool]:
    """Reference platform tools by their ``.name`` (declarative, no drift)."""
    return [RegistryTool(name=t.name) for t in tools]


CHECKOUT_CONFIG = AgentConfig(
    name="checkout",
    description="Guide the user through placing an order (identity → payment).",
    system_prompt=CHECKOUT_SYSTEM_PROMPT,
    # add_item is product_rec's job. Excluding it from the checkout leaf makes
    # it structurally impossible for the checkout subagent to RE-add an item
    # product_rec just added and handed off (which doubled cart quantities).
    # checkout keeps remove_item / set_quantity for mid-checkout adjustments.
    tools=_registry_tools([t for t in CHECKOUT_TOOLS if t.name != "add_item"]),
    skills=[SkillSpec(name=s["name"]) for s in CHECKOUT_SKILLS],
    middleware=[MiddlewareSpec(name="log_tool_calls")],
)

PRODUCT_REC_CONFIG = AgentConfig(
    name="product_rec",
    description="Browse + cart management: search, lookup, serviceability, add/remove/qty/view.",
    system_prompt=PRODUCT_REC_PROMPT,
    tools=_registry_tools(
        [
            search_products,
            get_product,
            check_serviceability,
            add_item,
            remove_item,
            set_quantity,
            get_cart_summary,
        ]
    ),
    middleware=[MiddlewareSpec(name="log_tool_calls")],
)

ORDER_STATUS_CONFIG = AgentConfig(
    name="order_status",
    description="Look up the status / tracking of a past order.",
    system_prompt=ORDER_STATUS_PROMPT,
    tools=_registry_tools([get_order_status, list_recent_orders]),
    middleware=[MiddlewareSpec(name="log_tool_calls")],
)

# =============================================================================
# Shared wrapper helpers (ported from v2 graph.py)
# =============================================================================
_PRODUCT_REC_HISTORY_TURNS = 8

# Matches lines produced by catalog_tools.search_products / get_product:
#   "P-2: Black Hoodie — $49.99 [apparel, hoodie, black]"
_PRODUCT_LINE_RE = re.compile(r"^(P-\d+):\s+(.+?)\s+[—\-]\s+\$(\S+)\s+\[(.+?)\]$")


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
    """Run a subagent via ``stream`` so ``log_tool_calls`` custom events reach
    the outer SSE/CLI. ``invoke()`` does not propagate custom stream chunks to
    the parent graph; streaming and re-emitting via ``get_stream_writer()``
    fixes missing SKILL/TOOL lines in the UI.
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


# =============================================================================
# Wrapper factories — close over the compiled leaf agent, return a graph node
# =============================================================================
def make_checkout_wrapper(agent: Any) -> Callable[[AgentState], Command]:
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
            agent,
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
            f"done step={cart.step.value} skills={result.get('skills_loaded', [])} "
            f"items={len(cart.items)}",
        )

        asks: list[str] = []
        if cart.step.value.startswith("collecting_"):
            asks = _asks_for_step(cart.step.value, cart)
        elif cart.ready_to_confirm() and not cart.confirmed:
            asks = ["explicit yes to place the order"]

        step_summary = (
            f"checkout subagent finished at step={cart.step.value}; items={len(cart.items)}"
        )

        sr = StepResult(
            sop=ids.CHECKOUT,
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

    return checkout_wrapper


def make_product_rec_wrapper(agent: Any) -> Callable[[AgentState], Command]:
    def product_rec_wrapper(state: AgentState) -> Command:
        """Run the product_rec subagent for one iteration.

        product_rec owns cart CONTENTS (add / remove / set quantity) plus
        browsing + serviceability. We diff the cart (product_id -> quantity)
        before/after to classify what happened: an add hands off to checkout;
        a remove or quantity decrease just reports the updated cart.
        """
        ctx = _runtime_context(state)
        before = {i.product_id: i.quantity for i in state.cart.items}

        # Pass recent conversation history so the subagent can resolve
        # pronouns like "them" / "those" to products it just presented.
        history = list(state.messages[-_PRODUCT_REC_HISTORY_TURNS:])
        if not history:
            history = [HumanMessage(content=state.last_user_message())]
        # Give the agent the current cart so it can EDIT it (resolve "the
        # hoodie" -> a product id, remove, change quantity) without searching
        # the catalog. Without this it tends to treat "remove the hoodie" as
        # a search.
        if state.cart.items:
            cart_note = (
                "Current cart: "
                + "; ".join(f"{i.product_id} {i.name} x{i.quantity}" for i in state.cart.items)
                + ". To edit it, use remove_item / set_quantity — do NOT search "
                "the catalog to remove or change an item already in the cart."
            )
            history = [SystemMessage(content=cart_note), *history]

        result = _stream_subagent(agent, {"messages": history}, context=ctx)

        products = _extract_products_from_messages(result["messages"])
        serviceability = _extract_serviceability_from_messages(result["messages"])
        viewed_cart = any(
            isinstance(m, ToolMessage) and getattr(m, "name", None) == "get_cart_summary"
            for m in result["messages"]
        )

        cart_now = ctx.cart_service.cart
        after = {i.product_id: i.quantity for i in cart_now.items}
        added = [pid for pid in after if after[pid] > before.get(pid, 0)]
        removed = [pid for pid in before if pid not in after]
        decreased = [pid for pid in after if pid in before and after[pid] < before[pid]]
        cart_changed = bool(added or removed or decreased)

        def _cart_lines() -> list[dict]:
            return [
                {"id": i.product_id, "name": i.name, "qty": i.quantity, "price": str(i.unit_price)}
                for i in cart_now.items
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
        elif viewed_cart and cart_now.items:
            # "what's in my cart" — get_cart_summary ran, nothing changed.
            summary = "showed the cart"
            details = {"cart_edit": {"removed": [], "decreased": [], "items": _cart_lines()}}
        else:
            summary = "no products matched the user's query"
            asks = ["clarify what you're looking for"]

        sr = StepResult(
            sop=ids.PRODUCT_REC,
            summary=summary,
            asks=asks,
            next_sop=next_sop,
            details=details,
            cart_diff={"items": len(cart_now.items)} if cart_changed else None,
        )
        return Command(
            goto="supervisor",
            update={"cart": cart_now, "step_results": [sr]},
        )

    return product_rec_wrapper


def make_order_status_wrapper(agent: Any) -> Callable[[AgentState], Command]:
    def order_status_wrapper(state: AgentState) -> Command:
        ctx = _runtime_context(state)
        result = _stream_subagent(
            agent,
            {"messages": [HumanMessage(content=state.last_user_message())]},
            context=ctx,
        )
        order_details = _extract_order_from_messages(result["messages"])
        sr = StepResult(
            sop=ids.ORDER_STATUS,
            summary=("looked up order status" if order_details else "could not find a matching order"),
            asks=[] if order_details else ["confirm the order id"],
            next_sop=None,
            details=order_details,
        )
        return Command(goto="supervisor", update={"step_results": [sr]})

    return order_status_wrapper


# =============================================================================
# Leaf registry — what the graph + supervisor are generated from
# =============================================================================
@dataclass(frozen=True)
class LeafSpec:
    """One leaf: its declarative agent config + its domain wrapper + routing.

    ``routing_help`` is the description block the supervisor injects into the
    classifier prompt for this leaf. ``wrapper_factory`` takes the compiled
    agent and returns the graph node fn.
    """

    name: str
    config: AgentConfig
    routing_help: str
    wrapper_factory: Callable[[Any], Callable[[AgentState], Command]]
    needs_checkpointer: bool = False
    needs_store: bool = True
    # The writer block kind this leaf produces (see agent_v4.output_schemas).
    # None = this leaf contributes only to the prose message, no typed block.
    output_block: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


LEAVES: list[LeafSpec] = [
    LeafSpec(
        name=ids.PRODUCT_REC,
        config=PRODUCT_REC_CONFIG,
        wrapper_factory=make_product_rec_wrapper,
        output_block="product_reco",
        routing_help=(
            "product_rec    Browsing AND cart management. Pre-purchase questions\n"
            "               (search the catalog, look up a product, delivery-area\n"
            '               questions like "do you ship to 94110?"), AND all cart\n'
            "               CONTENT edits: add an item, REMOVE an item, change a\n"
            '               quantity, plus "what is in my cart" / "why are there N".\n'
            "               Cart edits route here even mid-checkout. Adding an item\n"
            "               hands off to checkout."
        ),
    ),
    LeafSpec(
        name=ids.CHECKOUT,
        config=CHECKOUT_CONFIG,
        wrapper_factory=make_checkout_wrapper,
        needs_checkpointer=True,
        output_block="checkout",
        routing_help=(
            "checkout       The user is actively trying to buy: they want a cart\n"
            "               opened, or they're providing data the in-progress\n"
            "               checkout asked for (name, address, zip *as part of\n"
            '               checkout flow*, delivery option, payment, "yes" to\n'
            "               confirm)."
        ),
    ),
    LeafSpec(
        name=ids.ORDER_STATUS,
        config=ORDER_STATUS_CONFIG,
        wrapper_factory=make_order_status_wrapper,
        output_block="order_status",
        routing_help=(
            "order_status   The user is asking about a PAST order's status,\n"
            "               tracking, or delivery (order ids look like ORD-* or\n"
            "               RCPT-*)."
        ),
    ),
]

LEAF_NAMES: list[str] = [spec.name for spec in LEAVES]
LEAVES_BY_NAME: dict[str, LeafSpec] = {spec.name: spec for spec in LEAVES}

# Guard: the lightweight id constants and the registry must not drift.
assert set(LEAF_NAMES) == {ids.CHECKOUT, ids.PRODUCT_REC, ids.ORDER_STATUS}, (
    "agent_v4.ids constants and the LEAVES registry disagree"
)


def routing_catalog() -> str:
    """The per-leaf description block for the supervisor classifier prompt."""
    return "\n\n".join(f"  - {spec.routing_help}" for spec in LEAVES)


__all__ = [
    "LeafSpec",
    "LEAVES",
    "LEAF_NAMES",
    "LEAVES_BY_NAME",
    "routing_catalog",
    "CHECKOUT_CONFIG",
    "PRODUCT_REC_CONFIG",
    "ORDER_STATUS_CONFIG",
    "make_checkout_wrapper",
    "make_product_rec_wrapper",
    "make_order_status_wrapper",
]
