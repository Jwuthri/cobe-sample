"""The orchestrator (router) — the sole reader of the conversation.

It resolves the user's references into concrete ids, routes each distinct request to
exactly one worker, then stops. It never writes the customer-facing reply (the writer
does that). Three middleware primitives replace what Pydantic AI gave as natives:

  * **reference memo** — :func:`dynamic_instructions` injects ``build_memo`` (live cart
    + remembered facts) on every model call so it can resolve "the green one" without
    the workers seeing the chat;
  * **empty-cart guard** — :func:`hide_tool` drops the checkout delegate while the cart
    is empty, so an "add X" can never route to checkout. Structural, not prompt-driven;
  * **single tool per step** — :func:`no_parallel_tools` makes a compound message route
    ONE worker per model step, keeping the event bus ordered + the shared cart race-free.

The delegates (workers) are wired in at build time — they carry Python hooks, so they
are not declarative tools. ``build_orchestrator`` accepts injected worker agents so the
whole router can be driven by fake models offline (see the tests).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from langchain.tools import ToolRuntime
from langchain.agents import create_agent
from langchain_core.tools import tool

from lg_agent_v3.agents import checkout, order_status, product_rec
from lg_agent_v3.agents.names import CHECKOUT, ORDER_STATUS, PRODUCT_REC
from lg_agent_v3.runtime import (
    ShoppingDeps,
    Worker,
    build_model,
    compile_guardrails,
    dynamic_instructions,
    hide_tool,
    no_parallel_tools,
    run_subagent,
)

ROUTER_PROMPT = """\
You coordinate a shopping assistant with three sub-agent tools:

  - product_rec(query):  search the catalog, look up a product, answer
      delivery-area / serviceability questions ("do you ship to 94110?"), AND all
      cart-content edits (add an item, remove an item, change a quantity, "what's
      in my cart"). Adding an item is the cue to move to checkout next.
  - checkout(query):     drive an in-progress purchase — capture identity, address,
      delivery option, payment, and place the order ONLY on an explicit "yes".
      Needs items already in the cart.
  - order_status(query): look up a PAST order's status / tracking (ids look like
      ORD-* or RCPT-*).

How to route the user's latest message — handle EVERY distinct request in it,
calling one tool per request (a compound message like "find me a green cap and
check on order ORD-7" needs TWO tool calls):

  1. ANY browse / catalog / shopping intent -> call product_rec. This covers asking
     what's available ("what do you sell", "what products do you have", "show me
     your catalog", "do you have hats?"), asking about a product ("tell me about
     P-2"), searching ("find me a green cap"), serviceability ("do you ship to
     94110?"), AND adding to cart ("add X", "buy X", "I want X"). It works whether
     or not the cart is empty. (The checkout tool is unavailable while the cart is
     empty.)
  2. Cart NON-empty and the user is providing checkout data (their name, a shipping
     address, a delivery option, a payment method, a promo/discount code, or
     "yes"/"confirm" to a pending summary) -> call checkout.
  3. Cart edits or cart questions ("remove the hoodie", "make it 2", "what's in my
     cart") -> call product_rec, even mid-checkout. Checkout cannot add or remove
     items.
  4. Past-order tracking ("where's my order", an ORD-/RCPT- id) -> order_status.
  5. Smalltalk ONLY: greetings, thanks, off-topic chit-chat, or a question about
     what YOU (the assistant) can do / how you work ("what can you do", "who are
     you", "help") -> call NO tool. A question about PRODUCTS or the CATALOG is NOT
     smalltalk — it is rule 1 (product_rec).

Resolving references — THIS IS YOUR JOB (the sub-agents do NOT see the conversation;
they only get the ``query`` you write):
  - The user refers to things indirectly: "add it", "the green one", "the cheaper
    one", "the second", "that hoodie", "make it 2", "remove the cap". Resolve each to
    a CONCRETE product id (P-N) yourself, using the conversation plus the "Context
    for resolving references" block you're given (the current cart + the products
    most recently shown). Then pass a fully self-contained instruction that already
    names the id — e.g. "add P-4 to the cart", "set P-2 quantity to 3", "remove P-1".
    Never pass a bare "add it" or "the green one" to a sub-agent.
  - If a reference is genuinely ambiguous and the context doesn't pin it down, pass
    the user's description through (e.g. "search for a green cap") so product_rec can
    look it up. Never invent an id.
  - When the user references something established in an EARLIER turn — including a
    fact a DIFFERENT sub-agent surfaced — copy the relevant fact (an id OR a
    description) from the context into the query. NEVER assume a sub-agent saw the
    conversation: if a fact isn't in the query you write, the sub-agent does not
    know it.

Pass each tool a short, self-contained instruction as ``query``. Never invent a
product id, order id, or a request the user didn't make.

EXCEPTION — checkout data goes VERBATIM. When the user provides checkout data (a
name, an address, a delivery option, a payment method, or a yes/no), pass their
message EXACTLY as the ``query`` — do NOT prepend a field label like "Shipping
address:" and do NOT decide which field it is. The checkout agent maps it to the
right field from the cart's current step; a label you add can be mis-parsed AS the
data (e.g. the words "Shipping address" becoming the customer's name).

You do NOT write the customer-facing reply. As soon as every distinct request in the
user's message has been handled by a tool call — or the message was smalltalk that
needs no tool — reply with exactly the single word: DONE

A separate writer turns the tool results into the user's message. Do not summarize,
greet, or add any other text. Just route, then output DONE.
"""

# The workers, in routing-priority order.
WORKERS = [product_rec.WORKER, checkout.WORKER, order_status.WORKER]


def build_memo(deps: ShoppingDeps) -> str:
    """The reference-resolution context: live cart + facts remembered last turn.

    Two domain-agnostic sources, never the raw chat: the live structured state
    (``routing_context()``) and the per-step ``recall`` snippets a worker surfaced in a
    previous turn (kept in ``deps.routing_notes``). Returns ``""`` when there is nothing
    to resolve against. Used both as the orchestrator's dynamic instruction and
    (read-only) by the session's debug trace.
    """
    live = deps.routing_context()
    blocks = [text for text in live.values() if text]
    blocks += [text for key, text in deps.routing_notes.items() if key not in live and text]
    if not blocks:
        return ""
    return (
        "Context for resolving the user's references (authoritative — do NOT invent; "
        "if a reference is ambiguous, delegate to a sub-agent to look it up):\n"
        + "\n".join(blocks)
    )


def absorb_recalls(deps: ShoppingDeps) -> None:
    """Persist this turn's ``recall`` snippets so the NEXT turn's memo can use them."""
    for sr in deps.steps:
        if sr.recall:
            deps.routing_notes[sr.sop] = sr.recall


def _cart_empty(deps: ShoppingDeps) -> bool:
    return not deps.cart_service.cart.items


def _delegate_tools(workers: dict[str, Worker]) -> list:
    """Wrap each worker as an orchestrator tool — its name + description are the
    routing surface; its body is a thin call to :func:`run_subagent`."""

    @tool(PRODUCT_REC, description=product_rec.DESCRIPTION)
    async def _product_rec(query: str, runtime: ToolRuntime[ShoppingDeps]) -> str:
        return await run_subagent(runtime.context, workers[PRODUCT_REC], query)

    @tool(CHECKOUT, description=checkout.DESCRIPTION)
    async def _checkout(query: str, runtime: ToolRuntime[ShoppingDeps]) -> str:
        return await run_subagent(runtime.context, workers[CHECKOUT], query)

    @tool(ORDER_STATUS, description=order_status.DESCRIPTION)
    async def _order_status(query: str, runtime: ToolRuntime[ShoppingDeps]) -> str:
        return await run_subagent(runtime.context, workers[ORDER_STATUS], query)

    return [_product_rec, _checkout, _order_status]


def build_orchestrator(
    model: Any | None = None,
    worker_agents: dict[str, Any] | None = None,
    guardrails: list | None = None,
) -> Any:
    """Compile the router, with the sub-agents wired in as delegate tools.

    ``worker_agents`` (name → compiled sub-agent) lets tests swap in fake-model
    workers; omitted, each worker uses its module-level default agent. ``guardrails``
    are the router's own before/after_agent rules — a block routes the turn to the
    writer (the session reads the orchestrator hit → refusal).
    """
    workers = {w.name: w for w in WORKERS}
    for name, agent in (worker_agents or {}).items():
        workers[name] = replace(workers[name], agent=agent)

    return create_agent(
        model=model or build_model(0.0),
        tools=_delegate_tools(workers),
        system_prompt=ROUTER_PROMPT,
        context_schema=ShoppingDeps,
        middleware=[
            dynamic_instructions(build_memo),
            hide_tool(CHECKOUT, _cart_empty),
            no_parallel_tools(),
            *compile_guardrails(guardrails, "orchestrator"),
        ],
        name="orchestrator",
    )


orchestrator = build_orchestrator()
