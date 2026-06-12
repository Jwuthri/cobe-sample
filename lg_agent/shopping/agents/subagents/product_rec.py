"""The ``product_rec`` sub-agent — browse the catalog + manage the cart.

Everything that defines this on-the-fly agent lives in this one file: its prompt,
its JSON config (tools referenced by name), and the small Python hooks that bridge
its run to the shared cart + the writer:

  * ``snapshot``    — cart quantities before the run, for diffing what changed;
  * ``build_input`` — a self-contained query + a volatile cart note (no transcript);
  * ``extract``     — turn the run's tool results into a grounded ``StepResult``;
  * ``summarize``   — the one terse line the orchestrator LLM reads.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from lg_agent.core.step import StepResult
from lg_agent.core.subagent import SubAgent
from lg_agent.shopping.agents.subagents.names import CHECKOUT, PRODUCT_REC
from lg_agent.shopping.tools import PRODUCT_REC_TOOLS, registry_specs

MODEL = "openai:gpt-5.4-mini"

PROMPT = """\
You handle browsing AND cart management: search, look up products,
answer serviceability, and edit the cart (add, remove, change quantity,
and show what's in it).

You receive ONE self-contained instruction (you do NOT see the conversation).
The orchestrator has already resolved any reference like "the green one" / "add
it" into an explicit product id, so your instruction normally names the id
directly. The current cart is shown to you when relevant. Act on the instruction
via tool calls — don't ask for context you weren't given.

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

  * **"add P-3 to the cart", "add the green cap", "buy P-3"** → add_item with
    the product_id. Your instruction is self-contained and normally names the id
    already (the orchestrator resolved the reference). If it names a product by
    description but not an id, search_products first to find the id, then add_item.
    Product ids are case- and hyphen-insensitive ("p3", "P3", "p-3", "P-3" all
    mean P-3) — pass the canonical "P-N" form to add_item.

  * **"remove the hoodie", "remove P-2", "take the cap out", "I don't want
    the shoes anymore"** → this is a CART edit, NOT a search. Do NOT call
    search_products. The current cart contents are shown to you above (or
    call get_cart_summary); find the matching line and call
    remove_item(product_id) — or set_quantity(product_id, 0).

  * **"make it 2", "change the hoodie to 3", "I want two of those"** →
    set_quantity(product_id, qty). set_quantity(product_id, 0) removes it.

  * **"what's in my cart", "show my cart", "why are there 2 hoodies"** →
    get_cart_summary() and report the contents. Do NOT add or remove
    anything for a pure "what's in my cart" question.

## Rules

  - Never invent products. Only mention what the tools return.
  - Order ids (formats like ORD-123 or RCPT-9000) are NOT products.
    NEVER pass them to get_product or search_products — a separate agent
    handles order status.
  - If a search returns no matches, say so and ask the user to clarify.
  - When the instruction already names a product id to add, go straight to
    add_item — don't re-search to "confirm" it.
  - Be concise. The writer composes the final user-facing reply; you
    just do the work via tool calls.
"""

# The tool description is the orchestrator's routing surface — it decides whether
# to call this agent by reading this text.
DESCRIPTION = (
    "Search the catalog, look up a product, answer delivery-area questions, and "
    "edit the cart (add / remove / change quantity / show contents). Call this for "
    "ANY browsing or cart-content request. Pass a self-contained instruction as "
    "`query` (e.g. 'add P-2 to the cart', 'search for caps'). Adding an item is a "
    "natural cue to proceed to checkout next."
)

CONFIG = {
    "name": PRODUCT_REC,
    "description": "Browse + cart management: search, lookup, serviceability, add/remove/qty/view.",
    "system_prompt": PROMPT,
    "model": {"provider_model": MODEL, "temperature": 0.0},
    "tools": registry_specs(PRODUCT_REC_TOOLS),
    "middleware": [{"name": "log_tool_calls", "params": {"log_prefix": PRODUCT_REC}}],
}


# =============================================================================
# tool-result parsing
# =============================================================================
# Matches lines from search_products / get_product:
#   "P-2: Black Hoodie — $49.99 [apparel, hoodie, black]"
_PRODUCT_LINE_RE = re.compile(r"^(P-\d+):\s+(.+?)\s+[—\-]\s+\$(\S+)\s+\[(.+?)\]$")


def _parse_products(messages) -> list[dict]:
    """Parse search_products / get_product results into structured products."""
    products: list[dict] = []
    seen: set[str] = set()
    for m in messages:
        if not isinstance(m, ToolMessage) or getattr(m, "name", None) not in (
            "search_products",
            "get_product",
        ):
            continue
        for line in str(m.content).splitlines():
            match = _PRODUCT_LINE_RE.match(line.strip())
            if not match or match.group(1) in seen:
                continue
            seen.add(match.group(1))
            products.append(
                {
                    "id": match.group(1),
                    "name": match.group(2),
                    "price": match.group(3),
                    "tags": [t.strip() for t in match.group(4).split(",")],
                }
            )
    return products


def _parse_serviceability(messages) -> dict | None:
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "check_serviceability":
            content = str(m.content).strip()
            if content:
                return {"raw": content}
    return None


def _recall(products: list[dict]) -> str:
    """Domain-rendered note the orchestrator remembers next turn to resolve refs."""
    listed = "; ".join(
        f"{p['id']} {p['name']} ${p['price']}"
        + (f" [{', '.join(p.get('tags', []))}]" if p.get("tags") else "")
        for p in products
    )
    return (
        "Recently shown products (resolve references like 'the green one', 'it', "
        f"'the second one' to THESE exact ids): {listed}"
    )


# =============================================================================
# hooks
# =============================================================================
def snapshot(ctx) -> dict[str, int]:
    """Snapshot of {product_id: qty} before the run — for diffing what changed."""
    return {i.product_id: i.quantity for i in ctx.cart_service.cart.items}


def build_input(ctx, query: str) -> dict:
    """A volatile cart note (structured state) + the instruction. NO conversation.

    The orchestrator already resolved any reference into ``query``; the only ambient
    context the sub-agent needs is the live cart, so it can edit lines without
    re-searching.
    """
    cart = ctx.cart_service.cart
    messages = []
    if cart.items:
        note = (
            "Current cart: "
            + "; ".join(f"{i.product_id} {i.name} x{i.quantity}" for i in cart.items)
            + ". To edit it, use remove_item / set_quantity — do NOT search the "
            "catalog to remove or change an item already in the cart."
        )
        messages.append(SystemMessage(content=note))
    messages.append(HumanMessage(content=query))
    return {"messages": messages}


def extract(ctx, messages, before) -> StepResult:
    cart = ctx.cart_service.cart
    before = before or {}
    products = _parse_products(messages)
    serviceability = _parse_serviceability(messages)
    viewed_cart = any(getattr(m, "name", None) == "get_cart_summary" for m in messages)

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
        next_sop = CHECKOUT
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

    return StepResult(
        sop=PRODUCT_REC,
        summary=summary,
        asks=asks,
        next_sop=next_sop,
        details=details,
        cart_diff={"items": len(cart.items)} if cart_changed else None,
        recall=_recall(products) if products else None,
    )


def summarize(sr: StepResult, ctx) -> str:
    hint = " (you can proceed to checkout)" if sr.next_sop == CHECKOUT else ""
    return f"{sr.summary}{hint}"


SUBAGENT = SubAgent(
    name=PRODUCT_REC,
    description=DESCRIPTION,
    config=CONFIG,
    snapshot=snapshot,
    build_input=build_input,
    extract=extract,
    summarize=summarize,
    block="product_reco",
)

__all__ = ["SUBAGENT", "CONFIG", "PROMPT"]
