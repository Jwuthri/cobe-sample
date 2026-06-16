"""The ``product_rec`` worker — browse the catalog + manage the cart.

Everything that defines this worker lives here: its prompt, the compiled agent (with
its tools), a dynamic instruction that shows it the live cart, and the small hooks
that bridge a run to the shared cart + the writer:

  * ``_cart_note``  — a dynamic instruction injecting the current cart (so it can edit
    lines without re-searching);
  * ``_snapshot``   — cart quantities before the run, for diffing what changed;
  * ``extract``     — turn the run's tool outputs into a grounded ``StepResult``;
  * ``_summarize``  — the one terse line the orchestrator reads back.
"""

from __future__ import annotations

import re
from typing import Any

from langchain.agents import create_agent

from lg_agent_v2.agents import tools
from lg_agent_v2.agents.names import CHECKOUT, PRODUCT_REC
from lg_agent_v2.runtime import (
    ShoppingDeps,
    StepResult,
    Worker,
    build_model,
    dynamic_instructions,
    no_parallel_tools,
    tool_returns,
)

PROMPT = """\
You handle browsing AND cart management: search, look up products, answer
serviceability, and edit the cart (add, remove, change quantity, and show what's
in it).

You receive ONE self-contained instruction (you do NOT see the conversation). The
orchestrator has already resolved any reference like "the green one" / "add it"
into an explicit product id, so your instruction normally names the id directly.
The current cart is shown to you when relevant. Act on the instruction via tool
calls — don't ask for context you weren't given.

Your tools:
  - search_products(query, limit=5) — find products by free-text query.
                                       Empty query returns the full catalog.
  - get_product(product_id)         — fetch a single product by id.
  - check_serviceability(zip_code)  — does the store ship to a given zip, and with
                                       what options?
  - add_item(product_id, quantity=1) — add a product to the user's cart.
  - remove_item(product_id)          — remove a product line from the cart.
  - set_quantity(product_id, qty)    — set a line's quantity (0 removes it).
  - get_cart_summary()               — show the current cart + totals.

## Which tool to use

  * "do you sell X / what do you offer / show me the catalog / find me a Y"
    → search_products with the relevant query.

  * "tell me about P-2" → get_product("P-2").

  * "do you deliver to <place>", "do you ship to <place>", "do you serve <place>",
    "what shipping for 94110", "yes, <zip>" (after we just asked for zip) →
    check_serviceability. A "ship/deliver/serve to <place>" question is ALWAYS a
    serviceability check — NEVER search_products for the place name. If the user
    gave a city without a zip, check_serviceability will ask for the zip.

  * "add P-3 to the cart", "add the green cap", "buy P-3" → add_item with the
    product_id. Your instruction is self-contained and normally names the id
    already. If it names a product by description but not an id, search_products
    first to find the id, then add_item. Product ids are case- and
    hyphen-insensitive ("p3", "P3", "p-3", "P-3" all mean P-3).
    If the description matches MORE THAN ONE product (e.g. "add a hat" when there
    is a green AND a red cap), do NOT guess — list the matches with their ids and
    ask which one. Only add when there is a single clear match or an explicit id.

  * "remove the hoodie", "remove P-2", "take the cap out" → this is a CART edit,
    NOT a search. Do NOT call search_products. The current cart is shown to you
    (or call get_cart_summary); find the matching line and call remove_item — or
    set_quantity(product_id, 0).

  * "make it 2", "change the hoodie to 3" → set_quantity(product_id, qty).

  * "what's in my cart", "show my cart" → get_cart_summary() and report. Do NOT add
    or remove anything for a pure "what's in my cart" question.

## Rules

  - Never invent products. Only mention what the tools return.
  - Order ids (ORD-123, RCPT-9000) are NOT products. Never pass them to
    get_product or search_products — a separate agent handles order status.
  - When the instruction already names a product id to add, go straight to add_item
    — don't re-search to "confirm" it.
  - Be concise. The writer composes the final user-facing reply; you just do the
    work via tool calls. When you're done, reply with the single word DONE (an
    internal marker the user never sees) — always produce it, even if you took no
    action this turn.
"""

# The delegate-tool description is the orchestrator's routing surface.
DESCRIPTION = (
    "Search the catalog, look up a product, answer delivery-area questions, and edit "
    "the cart (add / remove / change quantity / show contents). Call this for ANY "
    "browsing or cart-content request. Pass a self-contained instruction as `query` "
    "(e.g. 'add P-2 to the cart', 'search for caps'). Adding an item is a natural cue "
    "to proceed to checkout next."
)


def _cart_note(deps: ShoppingDeps) -> str:
    """Show the worker the live cart so it can edit lines without re-searching."""
    cart = deps.cart_service.cart
    if not cart.items:
        return ""
    listed = "; ".join(f"{i.product_id} {i.name} x{i.quantity}" for i in cart.items)
    return (
        f"Current cart: {listed}. To edit it, use remove_item / set_quantity — do NOT "
        "search the catalog to remove or change an item already in the cart."
    )


def build(model: Any | None = None):
    """Compile the product_rec worker (optionally with an injected model, for tests)."""
    return create_agent(
        model=model or build_model(0.0),
        tools=[
            tools.search_products,
            tools.get_product,
            tools.check_serviceability,
            tools.add_item,
            tools.remove_item,
            tools.set_quantity,
            tools.get_cart_summary,
        ],
        system_prompt=PROMPT,
        context_schema=ShoppingDeps,
        # no_parallel_tools: cart edits mutate one shared list, so a compound "add X
        # and Y" must run one tool per step — parallel add_item races the cart AND
        # leaves an unanswered tool_call_id (provider 400). Reads can parallelize;
        # this worker can't.
        middleware=[dynamic_instructions(_cart_note), no_parallel_tools()],
        name=PRODUCT_REC,
    )


agent = build()


# =========================================================================== #
# tool-output parsing
# =========================================================================== #
# Matches a line from search_products / get_product:
#   "P-2: Black Hoodie — $49.99 [apparel, hoodie, black]"
_PRODUCT_LINE_RE = re.compile(r"^(P-\d+):\s+(.+?)\s+[—\-]\s+\$(\S+)\s+\[(.+?)\]$")


def _parse_products(returns) -> list[dict]:
    products: list[dict] = []
    seen: set[str] = set()
    for r in returns:
        if r.name not in ("search_products", "get_product"):
            continue
        for line in r.content.splitlines():
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


def _parse_serviceability(returns) -> dict | None:
    for r in reversed(returns):
        if r.name == "check_serviceability" and r.content.strip():
            return {"raw": r.content.strip()}
    return None


def _recall(products: list[dict]) -> str:
    listed = "; ".join(
        f"{p['id']} {p['name']} ${p['price']}" + (f" [{', '.join(p.get('tags', []))}]" if p.get("tags") else "")
        for p in products
    )
    return (
        "Recently shown products (resolve references like 'the green one', 'it', "
        f"'the second one' to THESE exact ids): {listed}"
    )


# =========================================================================== #
# hooks
# =========================================================================== #
def _snapshot(deps: ShoppingDeps) -> dict[str, int]:
    """{product_id: qty} before the run — for diffing what the worker changed."""
    return {i.product_id: i.quantity for i in deps.cart_service.cart.items}


def extract(deps: ShoppingDeps, messages, before) -> StepResult:
    cart = deps.cart_service.cart
    before = before or {}
    returns = tool_returns(messages)
    products = _parse_products(returns)
    serviceability = _parse_serviceability(returns)
    viewed_cart = any(r.name == "get_cart_summary" for r in returns)

    after = {i.product_id: i.quantity for i in cart.items}
    added = [pid for pid in after if after[pid] > before.get(pid, 0)]
    removed = [pid for pid in before if pid not in after]
    decreased = [pid for pid in after if pid in before and after[pid] < before[pid]]

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
        recall=_recall(products) if products else None,
    )


def _summarize(sr: StepResult, deps: ShoppingDeps) -> str:
    hint = " (you can proceed to checkout)" if sr.next_sop == CHECKOUT else ""
    return f"{sr.summary}{hint}"


WORKER = Worker(
    name=PRODUCT_REC,
    agent=agent,
    extract=extract,
    prompt=PROMPT,
    block="product_reco",
    snapshot=_snapshot,
    summarize=_summarize,
)
