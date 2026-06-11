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
You are the checkout assistant. You move ONE order forward.

Every turn you are given a "Checkout progress" block — the authoritative state of
the order (the cart persists every captured field across turns). Advance the
order as far as you can THIS turn:

  - Start from the first field that is not yet ✓ and go in order.
  - INTERNAL steps need no user input — always perform them when you reach them:
      * lookup_serviceability() right after an address is set,
      * quote_shipping() AND compute_tax() right after a delivery option is set.
  - STEPS THAT NEED THE USER — the delivery option, the payment method, and the
    final confirmation — use the user's LATEST message if it provides the answer.
    If the user's message does NOT provide it, STOP there and do nothing further
    (the writer will ask them). NEVER invent the user's choice.
  - NEVER re-capture a field already marked ✓ (don't re-call set_customer,
    set_address, set_delivery_option, or attach_payment for a ✓ field). Re-doing
    completed steps is the #1 mistake here — trust the progress block.

All checkout tools are already unlocked — call them directly (no load_skill).
Step → tool cheat-sheet:
  - identity:        set_customer(first_name, last_name, email?)
  - address:         set_address(street, city, zip_code, state?, country?)
  - serviceability:  lookup_serviceability()
  - delivery:        set_delivery_option(option) THEN quote_shipping() THEN compute_tax()
  - payment:         attach_payment(method, card_token?)   (card needs a token)
Call get_cart_summary() only if you genuinely need to double-check something.

Items are already in the cart from product selection — you have no add-item tool;
never try to add products.

## Confirmation rule (read carefully)

NEVER call ``confirm_checkout`` just because the cart is ready. Only call it when
the user's LATEST message is an explicit approval — "yes", "y", "confirm", "place
the order", "go ahead", "do it". If the cart is ready but the user hasn't said
yes, do NOTHING and stop — the writer will present the summary and ask. If the
user pushes back ("wait", "no", "actually change…"), handle that instead.

You don't speak to the user directly — the writer composes the reply. Do your
work via tool calls and stop.
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
# How many recent messages the *stateless* leaves (product_rec, order_status)
# see. The supervisor sees the FULL history (that's what fixes mis-routing), but
# the leaves only need recent context to resolve pronouns ("the green one",
# "them") — sending the whole transcript into every tool-loop call just inflates
# prompts (and, with human think-time between turns, the prompt cache has
# expired so that extra history is reprocessed uncached every turn). 16 messages
# ≈ 8 turns is plenty for reference resolution.
_SUBAGENT_HISTORY_MSGS = 16

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


# All checkout sub-skills, pre-unlocked every turn. The checkout subagent is now
# stateless + cart-anchored (see ``checkout_anchor``), so the old "load_skill in
# order" chaining — which made the model re-walk the whole flow each turn — is
# gone. Pre-unlocking lets it call the one tool the NEXT STEP needs directly.
ALL_CHECKOUT_SKILLS: list[str] = [s["name"] for s in CHECKOUT_SKILLS]

_NEXT_STEP_HINT = {
    "collecting_products": "items missing — this shouldn't happen mid-checkout.",
    "collecting_identity": "identity — capture the customer's name with set_customer.",
    "collecting_address": "address — capture the shipping address with set_address.",
    "awaiting_serviceability": "serviceability — call lookup_serviceability().",
    "collecting_delivery": "delivery — set_delivery_option the user chose, then quote_shipping() + compute_tax().",
    "collecting_payment": "payment — attach_payment with the user's method (card needs a token).",
    "awaiting_pricing": (
        "pricing — the cart changed, so the shipping quote and tax are stale. Recompute "
        "NOW yourself: call quote_shipping() then compute_tax(). Do NOT confirm yet — the "
        "refreshed total must be shown so the user can approve it."
    ),
    "ready_to_confirm": "ready — if the user's latest message is an explicit yes/confirm, call confirm_checkout(); otherwise do nothing.",
    "confirmed": "order already placed — do nothing.",
}


def checkout_anchor(cart) -> str:
    """Deterministic 'what's done / what's next' block injected each checkout turn.

    The cart is the source of truth, so we render its state explicitly instead of
    making the model rediscover it from a growing message thread (which caused it
    to re-execute completed steps). ``cart.step`` drives the single NEXT STEP.
    """
    c = cart

    def mark(done: bool, value: str) -> str:
        return f"✓ {value}".rstrip() if done else "— not provided"

    name = f"{c.customer.first_name or ''} {c.customer.last_name or ''}".strip()
    identity = mark(bool(c.customer.first_name), name)
    address = mark(
        c.address.is_complete(),
        f"{c.address.street}, {c.address.city} {c.address.zip_code}",
    )
    if c.serviceable is True:
        serviceability = f"✓ ships here (options: {', '.join(c.serviceable_options)})"
    elif c.serviceable is False:
        serviceability = "✗ NOT serviceable — ask for a different address"
    else:
        serviceability = "— not checked"
    delivery = mark(bool(c.delivery_option), c.delivery_option or "")
    payment = mark(bool(c.payment_method), c.payment_method or "")
    if c.shipping_is_fresh() and c.tax_is_fresh():
        pricing = f"✓ shipping {c.shipping.cost} + tax {c.tax.amount} → total {c.grand_total}"
    elif c.delivery_option:
        pricing = "✗ STALE — cart changed; recompute with quote_shipping() then compute_tax()"
    else:
        pricing = "— not computed"

    return (
        "Checkout progress (authoritative — never redo a ✓ field):\n"
        f"  identity:       {identity}\n"
        f"  address:        {address}\n"
        f"  serviceability: {serviceability}\n"
        f"  delivery:       {delivery}\n"
        f"  payment:        {payment}\n"
        f"  pricing:        {pricing}\n"
        f"Resume from: {_NEXT_STEP_HINT.get(c.step.value, 'the next missing field.')}\n"
        "Advance using the user's latest message + automatic internal steps; stop "
        "at the first field that needs info the user hasn't given."
    )


# =============================================================================
# Wrapper factories — close over the compiled leaf agent, return a graph node
# =============================================================================
def make_checkout_wrapper(agent: Any) -> Callable[[AgentState], Command]:
    def checkout_wrapper(state: AgentState) -> Command:
        """Run the checkout subagent for one iteration; return a StepResult."""
        ctx = _runtime_context(state)
        debug_log.graph(
            "checkout_wrapper",
            f"start step={state.cart.step.value} msg={state.last_user_message()[:100]!r}",
        )
        # Stateless + cart-anchored: inject the authoritative progress block and
        # pre-unlock all skills, so the subagent does only the next step instead
        # of re-walking the whole flow off a growing checkpointed thread.
        result = _stream_subagent(
            agent,
            {
                "messages": [
                    SystemMessage(content=checkout_anchor(state.cart)),
                    HumanMessage(content=state.last_user_message()),
                ],
                "skills_loaded": list(ALL_CHECKOUT_SKILLS),
            },
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

        # Pass recent conversation so the subagent can resolve pronouns
        # ("them" / "those" / "the cheaper one"). A bounded window (not the
        # whole transcript) keeps the prompt small — references are always
        # recent, and a smaller prompt is cheaper when the cross-turn cache
        # has expired.
        history = list(state.messages[-_SUBAGENT_HISTORY_MSGS:])
        if not history:
            history = [HumanMessage(content=state.last_user_message())]
        # Give the agent the current cart so it can EDIT it (resolve "the
        # hoodie" -> a product id, remove, change quantity) without searching
        # the catalog. APPEND it (don't prepend) so the stable history stays a
        # cacheable prefix — a volatile cart note at the FRONT would change the
        # prefix every turn and defeat OpenAI prompt caching.
        if state.cart.items:
            cart_note = (
                "Current cart: "
                + "; ".join(f"{i.product_id} {i.name} x{i.quantity}" for i in state.cart.items)
                + ". To edit it, use remove_item / set_quantity — do NOT search "
                "the catalog to remove or change an item already in the cart."
            )
            history = [*history, SystemMessage(content=cart_note)]

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
        # Recent conversation so follow-ups like "where's that one I asked
        # about?" resolve to an order id from a nearby turn. Bounded window —
        # the whole transcript isn't needed and just inflates the prompt.
        history = list(state.messages[-_SUBAGENT_HISTORY_MSGS:]) or [
            HumanMessage(content=state.last_user_message())
        ]
        result = _stream_subagent(
            agent,
            {"messages": history},
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
        # Stateless now: the cart (carried in AgentState / the shared context) is
        # the source of truth, and each turn re-anchors on it via checkout_anchor.
        # No checkpointed message thread → nothing for the model to re-walk.
        needs_checkpointer=False,
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
