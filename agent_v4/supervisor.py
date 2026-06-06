"""Supervisor — bounded loop, conversation-aware.

Two responsibilities per call:
  1. Decide whether the turn is **done** (route to writer) or whether
     another SOP needs to run.
  2. If not done, pick the next SOP — using either an explicit
     ``next_sop`` hint from the previous step result, or a fresh
     OpenAI classification that sees the conversation history + the
     accumulated step results + the current cart step.

``MAX_ITERATIONS`` caps the loop so a bad classifier or recursive
hand-off can't burn through tokens.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from agent_v4.checkout.cart import CheckoutStep
from agent_v4.llm import classifier_client, model_name
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agent_v4.state import AgentState
    from agent_v4.step_result import StepResult


MAX_ITERATIONS = 4
HISTORY_TURNS = 8


class SOPName(str, Enum):
    CHECKOUT = "checkout"
    ORDER_STATUS = "order_status"
    PRODUCT_REC = "product_rec"


class SupervisorDecision(BaseModel):
    """Structured-output shape returned by the classifier."""

    done: bool = Field(description="True when no more SOP work is needed this turn.")
    next_sop: SOPName | None = Field(
        default=None, description="Required when done=False; ignored otherwise."
    )
    reason: str = Field(default="", description="One sentence justification.")


_CLASSIFIER_PROMPT = """\
You are routing the latest user message in a multi-agent shopping
assistant. Decide whether more SOP work is needed this turn and, if
so, which SOP to call next.

The three SOPs and what they DO:

  - product_rec    Pre-purchase questions: search the catalog, look up
                   a single product, AND answer delivery-area questions
                   ("do you ship to <city/zip>?", "what shipping is
                   available in 94110?"). Use this for ANY question
                   that doesn't require a cart.
  - checkout       The user is actively trying to buy: they want a cart
                   opened, or they're providing data the in-progress
                   checkout asked for (name, address, zip *as part of
                   checkout flow*, delivery option, payment, "yes" to
                   confirm).
  - order_status   The user is asking about a PAST order's status,
                   tracking, or delivery (order ids look like ORD-* or
                   RCPT-*).

When to set done=True (no SOP work needed):
  - Smalltalk / greetings / off-topic / vague chatter ("hi", "thanks",
    "lol", "what can you do"). The writer will reply conversationally.
  - The most recent step_result has non-empty ``asks`` — the user must
    respond before any more SOP work makes sense.

How to pick a SOP — read CAREFULLY:

  1) **Empty cart → product_rec by default.** If ``cart_step`` is
     ``collecting_products`` (cart has no items), ANY shopping intent
     — including "add X to my cart", "buy X", "I want X", "get me
     X" — goes to **product_rec**. Product_rec will identify the
     product (via search_products or check_serviceability), add it
     to the cart, and hand off to checkout via ``next_sop``. NEVER
     route to checkout from an empty cart for an "add to cart"
     request: the checkout SOP has nothing meaningful to do without
     items.

  2) **Mid-checkout data provision → checkout.** If the cart is
     NON-empty (cart_step is past collecting_products) AND the user
     is providing data the checkout flow is currently asking for
     (their name, a shipping address as part of a buy, a delivery
     option, a payment method, "yes" / "y" / "confirm" to a pending
     order summary) → checkout.

  3) **Mid-checkout escape for browse questions.** If the user is
     mid-checkout BUT just asked a generic pre-purchase question
     that doesn't move checkout forward — "what do you offer",
     "what shoes do you sell", "do you deliver to X", "show me
     other options", "what's your return policy" — route to
     **product_rec**, NOT checkout. Don't trap the user in checkout
     for browse-style questions.

  4) **Serviceability questions** ("do you deliver to <city/zip>?",
     "what shipping options for <zip>?") with NO active checkout
     context → **product_rec** (it has check_serviceability).

  5) **Compound asks** like "find me a green cap and pay" →
     product_rec first; checkout is queued via ``next_sop``.

  6) **Past-order tracking** ("where's my order ORD-7?",
     "RCPT-9000 status?") → order_status.

Never invent a SOP. Default to done=True for anything that doesn't
clearly match one of the three.
"""


_SMALLTALK_KEYWORDS = {
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "thanks",
    "thank you",
    "ty",
    "ok",
    "okay",
    "cool",
    "nice",
    "lol",
    "bye",
    "goodbye",
    "see you",
}


def _is_likely_smalltalk(user_msg: str) -> bool:
    """Cheap heuristic to skip the LLM classifier for obvious greetings.

    Conservative: only triggers when the message is short AND every
    significant token is a known greeting/smalltalk word. Real shopping
    intents (even short ones like "buy P-1") fall through to the LLM.
    """
    msg = user_msg.strip().lower().rstrip("!?.")
    if not msg or len(msg) > 40:
        return False
    if msg in _SMALLTALK_KEYWORDS:
        return True
    # "hello how are you", "hi there"
    words = [w for w in msg.replace(",", " ").split() if w not in {"there", "you", "are", "how"}]
    return bool(words) and all(w in _SMALLTALK_KEYWORDS for w in words)


def _format_history(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"USER: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"ASSISTANT: {m.content}")
    return "\n".join(lines) if lines else "(no prior turns)"


def _format_step_results(results: list["StepResult"]) -> str:
    if not results:
        return "(none)"
    return "\n".join(
        f"- {r.sop.value}: {r.summary}"
        + (f" asks={r.asks}" if r.asks else "")
        + (f" next_sop={r.next_sop.value}" if r.next_sop else "")
        for r in results
    )


def classify_with_history(state: "AgentState") -> SupervisorDecision:
    history = state.messages[-HISTORY_TURNS:]
    user_payload = (
        f"Recent conversation:\n{_format_history(history)}\n\n"
        f"Cart step: {state.cart.step.value}\n"
        f"Step results this turn:\n{_format_step_results(state.step_results)}\n\n"
        "Decide done + next_sop."
    )
    client = classifier_client()
    resp = client.chat.completions.parse(
        model=model_name(),
        messages=[
            {"role": "system", "content": _CLASSIFIER_PROMPT},
            {"role": "user", "content": user_payload},
        ],
        response_format=SupervisorDecision,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        return SupervisorDecision(done=False, next_sop=SOPName.PRODUCT_REC, reason="fallback")
    if not parsed.done and parsed.next_sop is None:
        # Defensive: classifier said not-done but didn't pick a SOP.
        parsed = parsed.model_copy(update={"next_sop": SOPName.PRODUCT_REC})
    return parsed


def supervisor(state: "AgentState") -> Command:
    """Outer-graph supervisor node. Routes to a SOP wrapper or to the writer."""
    # 1. Hard cap: never loop forever.
    if state.iteration >= MAX_ITERATIONS:
        return Command(goto="writer", update={"iteration": 0})

    # 2. If the most recent step explicitly hints a next SOP, follow it.
    #    This is the cheap fast-path for compound asks
    #    (product_rec → checkout) and similar planned hand-offs.
    if state.step_results:
        last = state.step_results[-1]
        if last.next_sop is not None:
            return Command(
                goto=f"{last.next_sop.value}_wrapper",
                update={
                    "active_sop": last.next_sop,
                    "iteration": state.iteration + 1,
                },
            )
        # 3. The last SOP has nothing more to do AND is waiting on the
        #    user (it surfaced `asks`). No point running another SOP —
        #    user must respond first. Send to writer.
        if last.asks:
            return Command(goto="writer", update={"iteration": 0})

    # 4. Smalltalk shortcut: skip the LLM classifier on obvious
    #    greetings / off-topic chatter so we don't run a SOP for "hi".
    if not state.step_results and _is_likely_smalltalk(state.last_user_message()):
        return Command(goto="writer", update={"iteration": 0})

    # 5. Classify with full context (recent history + step results +
    #    cart step).
    decision = classify_with_history(state)
    if decision.done:
        return Command(goto="writer", update={"iteration": 0})
    assert decision.next_sop is not None  # narrowed by classify_with_history

    # 5b. Empty-cart safety net.
    #
    #     "add the cap to my cart" / "buy X" / "I want X" sometimes
    #     trips the classifier into picking ``checkout`` because of the
    #     word "cart"/"buy". But checkout has nothing useful to do with
    #     an empty cart — the catalog needs to find the product first.
    #
    #     We override the decision here. Same defense-in-depth pattern
    #     as ``checkout_gate``: trust the classifier 95% of the time,
    #     but never let it pick a SOP that's structurally unable to
    #     make progress. Skipped when we already ran product_rec this
    #     turn (the rerun guard below will route to writer).
    if (
        decision.next_sop == SOPName.CHECKOUT
        and state.cart.step == CheckoutStep.COLLECTING_PRODUCTS
        and SOPName.PRODUCT_REC not in {r.sop for r in state.step_results}
    ):
        decision = decision.model_copy(
            update={
                "next_sop": SOPName.PRODUCT_REC,
                "reason": f"empty-cart override (was: {decision.reason or 'checkout'})",
            }
        )

    # 6. Don't re-enter the same SOP we already ran this turn. If the
    #    classifier insists on the same SOP a second time, the SOP has
    #    already produced its step result — treat as done.
    already_ran = {r.sop for r in state.step_results}
    if decision.next_sop in already_ran:
        return Command(goto="writer", update={"iteration": 0})

    return Command(
        goto=f"{decision.next_sop.value}_wrapper",
        update={
            "active_sop": decision.next_sop,
            "iteration": state.iteration + 1,
        },
    )


# Kept for back-compat with v3 imports — tests / external callers can
# still ``from agent_v4.supervisor import classify``.
def classify(user_msg: str) -> SOPName:
    """Legacy one-shot classifier (no history). Avoid in new code."""
    if not user_msg.strip():
        return SOPName.PRODUCT_REC
    client = classifier_client()
    resp = client.chat.completions.parse(
        model=model_name(),
        messages=[
            {"role": "system", "content": _CLASSIFIER_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=SupervisorDecision,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None or parsed.next_sop is None:
        return SOPName.PRODUCT_REC
    return parsed.next_sop
