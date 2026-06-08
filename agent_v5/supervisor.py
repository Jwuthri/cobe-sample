"""The single supervisor agent — built two ways for the A/B comparison.

Both variants are ONE ``create_agent`` whose tools are the three subagent
wrappers from :mod:`agent_v5.subagents`. They share:
  * the same routing policy (ported from v4's classifier prompt),
  * the empty-cart guard (``wrap_model_call``) + the loop cap
    (``ToolCallLimitMiddleware`` — v4's ``MAX_ITERATIONS``),
  * the same ``context_schema`` so the live cart flows to every tool.

They differ in ONE thing — who writes the user-facing prose:

  * ``variant="speaking"`` (NO separate writer): the supervisor's own final
    message IS the reply. One agent does routing + voice.
  * ``variant="router"`` (WITH a writer): the supervisor only calls tools and
    then emits the sentinel ``DONE``; :mod:`agent_v5.writer` composes the reply
    from the accumulated step results + cart (the v4 design).

That single difference is exactly what the eval measures.
"""

from __future__ import annotations

from typing import Literal

from agent_v4.llm import chat_model
from agent_v5.middleware import empty_cart_guard
from agent_v5.subagents import SUBAGENT_TOOLS
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

Variant = Literal["speaking", "router"]

# v4's MAX_ITERATIONS = 4: at most four subagent calls per turn, then stop.
MAX_TOOL_CALLS = 4


# Routing policy — lifted from agent_v4/supervisor.py's classifier prompt, with
# "route to leaf X" rewritten as "call tool X".
_ROUTING_CORE = """\
You coordinate a shopping assistant with three subagent tools:

  - product_rec(query):  search the catalog, look up a product, answer
      delivery-area / serviceability questions ("do you ship to 94110?"), AND all
      cart-content edits (add an item, remove an item, change a quantity, "what's
      in my cart"). Adding an item is the cue to move to checkout next.
  - checkout(query):     drive an in-progress purchase — capture identity,
      address, delivery option, payment, and place the order ONLY on an explicit
      "yes". Needs items already in the cart.
  - order_status(query): look up a PAST order's status / tracking (ids look like
      ORD-* or RCPT-*).

How to route the user's latest message — handle EVERY distinct request in it,
calling one tool per request (a compound message like "find me a green cap and
check on order ORD-7" needs TWO tool calls):

  1. Empty cart + any shopping intent ("add X", "buy X", "I want X") -> call
     product_rec. It identifies the product, adds it, and signals checkout next.
     (The checkout tool is unavailable while the cart is empty.)
  2. Cart NON-empty and the user is providing checkout data (their name, a
     shipping address, a delivery option, a payment method, or "yes"/"confirm"
     to a pending summary) -> call checkout.
  3. Cart edits or cart questions ("remove the hoodie", "make it 2", "what's in
     my cart") -> call product_rec, even mid-checkout. Checkout cannot add or
     remove items.
  4. Generic pre-purchase / browse questions mid-checkout ("what else do you
     sell", "do you deliver to X") -> product_rec, not checkout.
  5. Past-order tracking -> order_status.
  6. Smalltalk / greetings / off-topic / "what can you do" -> call NO tool.

Pass each tool a short, self-contained instruction as ``query``. Never invent a
product id, order id, or a request the user didn't make."""


# Variant B — the supervisor speaks. Routing + the v4 writer's voice invariants.
_SPEAKING_TAIL = """\

After the needed tool(s) have run, write ONE clear, concise, friendly reply to
the user. Voice rules (these are strict):

  - Smalltalk turns (no tool was needed): reply briefly and warmly. Do NOT
    mention the cart, items, checkout, or addresses unless the user asked. If
    asked what you can do, say in one line: find products, place orders, check
    order status.
  - Use the tool results as your source of truth. Product lists, order details,
    and the checkout summary are rendered to the user as separate cards — copy
    ids verbatim when you must reference them, but introduce them naturally
    ("Here are the caps:") instead of re-dumping every id and price.
  - If a checkout tool result lists what's needed next, ask the user for exactly
    those things. Don't re-ask for anything they already provided earlier in the
    conversation.
  - NEVER say or imply the order is placed, confirmed, paid, or on its way unless
    a checkout tool result explicitly reported "ORDER CONFIRMED". If it didn't,
    only summarize the cart and ask for what's missing or for a "yes" — even if
    the user just said "confirm".
  - Friendly but brief. No emoji unless the user used one."""


# Variant A — the supervisor only routes; a separate writer composes the reply.
_ROUTER_TAIL = """\

You do NOT write the customer-facing reply. As soon as every distinct request in
the user's message has been handled by a tool call — or the message was
smalltalk that needs no tool — respond with exactly the single word:

  DONE

A separate writer turns the tool results into the user's message. Do not add any
other text, do not summarize, do not greet. Just route, then output DONE."""


SUPERVISOR_PROMPT_SPEAKING = _ROUTING_CORE + _SPEAKING_TAIL
SUPERVISOR_PROMPT_ROUTER = _ROUTING_CORE + _ROUTER_TAIL


def build_supervisor(variant: Variant):
    """Compile the supervisor agent for the given variant.

    Same tools / guard / cap / context for both; only the system prompt changes.
    """
    from agent_v5.context import SupervisorContext

    system_prompt = (
        SUPERVISOR_PROMPT_SPEAKING if variant == "speaking" else SUPERVISOR_PROMPT_ROUTER
    )
    return create_agent(
        model=chat_model(temperature=0.3 if variant == "speaking" else 0.0),
        tools=SUBAGENT_TOOLS,
        system_prompt=system_prompt,
        context_schema=SupervisorContext,
        middleware=[
            empty_cart_guard,
            ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS, exit_behavior="end"),
        ],
        name=f"supervisor_{variant}",
    )


__all__ = [
    "build_supervisor",
    "Variant",
    "MAX_TOOL_CALLS",
    "SUPERVISOR_PROMPT_SPEAKING",
    "SUPERVISOR_PROMPT_ROUTER",
]
