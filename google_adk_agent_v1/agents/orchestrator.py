"""The orchestrator (router) — the sole reader of the conversation.

It resolves the user's references into concrete ids, routes each distinct request to
exactly one worker, then stops. It never writes the customer-facing reply (the
writer does that). Two ADK features replace what used to be bespoke middleware:

  * **reference memo** — an instruction provider (live cart + remembered facts) so it
    can resolve "the green one" without the workers seeing the chat;
  * **empty-cart guard** — a ``before_model_callback`` strips the ``checkout`` tool's
    declaration from the request while the cart is empty, so an "add X" can never
    route to checkout. Structural, not prompt-enforced.

The delegate tools ARE the workers: each is a thin async function that calls
:func:`~google_adk_agent_v1.runtime.delegation.run_subagent`. ``deps.lock`` (held
inside that wrapper) makes a compound message route ONE worker at a time, keeping the
event stream ordered and the shared cart free of races.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LlmRequest
from google.adk.tools import ToolContext

from google_adk_agent_v1.agents import checkout as _checkout_mod
from google_adk_agent_v1.agents import order_status as _order_status_mod
from google_adk_agent_v1.agents import product_rec as _product_rec_mod
from google_adk_agent_v1.agents.names import CHECKOUT, ORDER_STATUS, PRODUCT_REC
from google_adk_agent_v1.runtime import ShoppingDeps, gen_config, make_model, run_subagent
from google_adk_agent_v1.runtime import registry

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


# --------------------------------------------------------------------------- #
# the three delegate tools (the orchestrator's "tools" ARE the workers).
# Each function's NAME is the tool name the prompt routes to; its docstring is the
# routing-surface description the model reads when choosing.
# --------------------------------------------------------------------------- #
async def product_rec(tool_context: ToolContext, query: str) -> str:
    return await run_subagent(tool_context, _product_rec_mod.WORKER, query)


async def checkout(tool_context: ToolContext, query: str) -> str:
    return await run_subagent(tool_context, _checkout_mod.WORKER, query)


async def order_status(tool_context: ToolContext, query: str) -> str:
    return await run_subagent(tool_context, _order_status_mod.WORKER, query)


product_rec.__doc__ = _product_rec_mod.DESCRIPTION
checkout.__doc__ = _checkout_mod.DESCRIPTION
order_status.__doc__ = _order_status_mod.DESCRIPTION


def _hide_checkout_when_cart_empty(callback_context: CallbackContext, llm_request: LlmRequest):
    """The empty-cart guard: strip the checkout tool's declaration while cart is empty.

    Returning ``None`` lets the (now checkout-free) request proceed to the model.
    """
    key = callback_context.state.get(registry.RUNTIME_KEY)
    if key is None:
        return None
    if registry.get(key).cart_service.cart.items:
        return None  # cart has items — leave all tools in place
    for tool in (llm_request.config.tools or []):
        decls = getattr(tool, "function_declarations", None)
        if decls:
            tool.function_declarations = [d for d in decls if d.name != CHECKOUT]
    return None


def build_memo(deps: ShoppingDeps) -> str:
    """The reference-resolution context: live cart + facts remembered last turn.

    Two domain-agnostic sources, never the raw chat: the live structured state
    (``routing_context()``) and the per-step ``recall`` snippets a worker surfaced
    in a previous turn (kept in ``deps.routing_notes``). Returns ``""`` when there is
    nothing to resolve against. Used both as the orchestrator's dynamic instruction
    and (read-only) by the session's debug trace.
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


def _routing_memo(ctx: ReadonlyContext) -> str:
    """Instruction provider: the static router prompt + the live reference memo."""
    key = ctx.state.get(registry.RUNTIME_KEY)
    if key is None:
        return ROUTER_PROMPT
    memo = build_memo(registry.get(key))
    return ROUTER_PROMPT + ("\n\n" + memo if memo else "")


orchestrator = LlmAgent(
    name="orchestrator",
    model=make_model(),
    generate_content_config=gen_config(0.0),
    instruction=_routing_memo,
    tools=[product_rec, checkout, order_status],
    before_model_callback=_hide_checkout_when_cart_empty,
)


def absorb_recalls(deps: ShoppingDeps) -> None:
    """Persist this turn's ``recall`` snippets so the NEXT turn's memo can use them."""
    for sr in deps.steps:
        if sr.recall:
            deps.routing_notes[sr.sop] = sr.recall
