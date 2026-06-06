"""Supervisor — bounded loop, conversation-aware, data-driven over LEAVES.

Two responsibilities per call:
  1. Decide whether the turn is **done** (route to writer) or whether
     another leaf needs to run.
  2. If not done, pick the next leaf — using either an explicit
     ``next_sop`` hint from the previous step result, or a fresh
     OpenAI classification that sees the conversation history + the
     accumulated step results + the current cart step.

The *topology* (which leaves exist, what the classifier may pick) comes
from :data:`agent_v4.leaves.LEAVES`; the *routing policy* (the prose and
the empty-cart / re-run / smalltalk guards) is domain code that names
specific leaves via :mod:`agent_v4.ids`. ``MAX_ITERATIONS`` caps the loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_v4 import ids
from agent_v4.checkout.cart import CheckoutStep
from agent_v4.leaves import LEAF_NAMES, routing_catalog
from agent_v4.llm import classifier_client, model_name
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agent_v4.state import AgentState
    from agent_v4.step_result import StepResult


MAX_ITERATIONS = 4
HISTORY_TURNS = 8


class SupervisorDecision(BaseModel):
    """Structured-output shape returned by the classifier.

    ``next_sop`` is a leaf id; it's validated against the live leaf set
    after parsing (see :func:`classify_with_history`).
    """

    done: bool = Field(description="True when no more leaf work is needed this turn.")
    next_sop: str | None = Field(
        default=None, description="Required when done=False; one of the valid leaf ids."
    )
    reason: str = Field(default="", description="One sentence justification.")


_CLASSIFIER_PROMPT = f"""\
You are routing the latest user message in a multi-agent shopping
assistant. You are called REPEATEDLY within a single turn: each call you
can see the steps already run this turn and their results (under "Step
results this turn"). Decide whether more leaf work is needed and, if so,
which leaf to call next — handle ONE not-yet-handled part of the user's
message per call, and use prior step results to inform the choice.

The leaves and what they DO:

{routing_catalog()}

When to set done=True (no MORE leaf work this turn):
  - Smalltalk / greetings / off-topic / vague chatter ("hi", "thanks",
    "lol", "what can you do"). The writer will reply conversationally.
  - EVERY distinct request in the user's latest message has already been
    handled by a step this turn. Check "Step results this turn": if the
    user asked for TWO things (e.g. a product question AND an order-status
    question) and only one of them has a step so far, you are NOT done —
    pick the leaf for the OTHER part.

  IMPORTANT: a leaf having asked the user a follow-up question (its
  ``asks`` is non-empty) does NOT by itself mean the turn is done — it
  only means that one leaf is now waiting on the user. If a DIFFERENT
  part of the message is still unhandled, route to its leaf now. Only set
  done=True once nothing in the message is left to handle. Never invent a
  request the user didn't make.

How to pick a leaf — read CAREFULLY:

  1) **Empty cart -> product_rec by default.** If ``cart_step`` is
     ``collecting_products`` (cart has no items), ANY shopping intent
     — including "add X to my cart", "buy X", "I want X", "get me
     X" — goes to **product_rec**. Product_rec will identify the
     product (via search_products or check_serviceability), add it
     to the cart, and hand off to checkout via ``next_sop``. NEVER
     route to checkout from an empty cart for an "add to cart"
     request: the checkout leaf has nothing meaningful to do without
     items.

  2) **Mid-checkout data provision -> checkout.** If the cart is
     NON-empty (cart_step is past collecting_products) AND the user
     is providing data the checkout flow is currently asking for
     (their name, a shipping address as part of a buy, a delivery
     option, a payment method, "yes" / "y" / "confirm" to a pending
     order summary) -> checkout.

  3) **Mid-checkout escape for browse questions.** If the user is
     mid-checkout BUT just asked a generic pre-purchase question
     that doesn't move checkout forward — "what do you offer",
     "what shoes do you sell", "do you deliver to X", "show me
     other options", "what's your return policy" — route to
     **product_rec**, NOT checkout. Don't trap the user in checkout
     for browse-style questions.

  3b) **Cart edits / cart questions -> product_rec, even mid-checkout.**
     "remove the hoodie", "take P-2 out", "change the quantity to 2",
     "make it 1", "what's in my cart", "why are there 2 hoodies" — these
     edit or inspect the CART CONTENTS and belong to product_rec (it owns
     add / remove / quantity / view). The checkout leaf is fulfillment
     only and cannot add or remove products.

  4) **Serviceability questions** ("do you deliver to <city/zip>?",
     "what shipping options for <zip>?") with NO active checkout
     context -> **product_rec** (it has check_serviceability).

  5) **Compound asks** like "find me a green cap and pay" ->
     product_rec first; checkout is queued via ``next_sop``.

  6) **Past-order tracking** ("where's my order ORD-7?",
     "RCPT-9000 status?") -> order_status.

Valid next_sop values: {", ".join(LEAF_NAMES)}.
Never invent a leaf. Default to done=True for anything that doesn't
clearly match one of the leaves.
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
        f"- {r.sop}: {r.summary}"
        + (f" asks={r.asks}" if r.asks else "")
        + (f" next_sop={r.next_sop}" if r.next_sop else "")
        for r in results
    )


def _coerce_next_sop(value: str | None) -> str | None:
    """Validate the classifier's pick against the live leaf set."""
    if value and value in LEAF_NAMES:
        return value
    return None


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
        return SupervisorDecision(done=False, next_sop=ids.DEFAULT_SOP, reason="fallback")
    next_sop = _coerce_next_sop(parsed.next_sop)
    if not parsed.done and next_sop is None:
        # Classifier said not-done but didn't name a valid leaf.
        next_sop = ids.DEFAULT_SOP
    return parsed.model_copy(update={"next_sop": next_sop})


def supervisor(state: "AgentState") -> Command:
    """Outer-graph supervisor node. Routes to a leaf wrapper or to the writer."""
    # 1. Hard cap: never loop forever.
    if state.iteration >= MAX_ITERATIONS:
        return Command(goto="writer", update={"iteration": 0})

    # 2. If the most recent step explicitly hints a next leaf, follow it.
    #    Deterministic output-aware handoff (e.g. product_rec added an item
    #    -> checkout). This is the coded version of "use a leaf's output to
    #    decide where to look next".
    if state.step_results:
        last = state.step_results[-1]
        if last.next_sop is not None and last.next_sop not in {r.sop for r in state.step_results[:-1]}:
            return Command(
                goto=f"{last.next_sop}_wrapper",
                update={
                    "active_sop": last.next_sop,
                    "iteration": state.iteration + 1,
                },
            )

    # 3. A leaf returning `asks` no longer ends the turn: the classifier is
    #    re-consulted (it sees the prior step results + their outputs) so a
    #    compound message's OTHER intents still get routed. The classifier's
    #    "done" rule decides when nothing is left to handle, and the re-run
    #    guard (step 6) + MAX_ITERATIONS keep the loop finite.

    # 4. Smalltalk shortcut: skip the LLM classifier on obvious greetings.
    if not state.step_results and _is_likely_smalltalk(state.last_user_message()):
        return Command(goto="writer", update={"iteration": 0})

    # 5. Classify with full context.
    decision = classify_with_history(state)
    if decision.done:
        return Command(goto="writer", update={"iteration": 0})
    next_sop = decision.next_sop or ids.DEFAULT_SOP

    # 5b. Empty-cart safety net: never let the classifier pick a leaf that's
    #     structurally unable to make progress (checkout with an empty cart).
    if (
        next_sop == ids.CHECKOUT
        and state.cart.step == CheckoutStep.COLLECTING_PRODUCTS
        and ids.PRODUCT_REC not in {r.sop for r in state.step_results}
    ):
        next_sop = ids.PRODUCT_REC

    # 6. Don't re-enter the same leaf we already ran this turn.
    already_ran = {r.sop for r in state.step_results}
    if next_sop in already_ran:
        return Command(goto="writer", update={"iteration": 0})

    return Command(
        goto=f"{next_sop}_wrapper",
        update={
            "active_sop": next_sop,
            "iteration": state.iteration + 1,
        },
    )


# Kept for back-compat with callers that imported a one-shot classifier.
def classify(user_msg: str) -> str:
    """Legacy one-shot classifier (no history). Returns a leaf id."""
    if not user_msg.strip():
        return ids.DEFAULT_SOP
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
    if parsed is None:
        return ids.DEFAULT_SOP
    return _coerce_next_sop(parsed.next_sop) or ids.DEFAULT_SOP
