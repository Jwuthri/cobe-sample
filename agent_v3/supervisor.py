"""Supervisor — bounded, conversation-aware routing (Agno port).

v2 was a LangGraph node returning a ``Command(goto=...)``. v3 splits it:

  - ``classify_with_history`` runs an Agno ``Agent`` with
    ``output_schema=SupervisorDecision`` (replacing the bare
    ``openai.OpenAI().chat.completions.parse`` call).
  - ``supervisor_selector`` is the Agno **Router** selector — it reads the
    shared ``session_state`` and returns the name of the next step to run:
    one of the SOP steps (``checkout`` / ``product_rec`` / ``order_status``)
    or the terminal ``finalize`` step that ends the loop (its content is
    the sentinel the loop's ``end_condition`` watches for).

The routing heuristics (smalltalk shortcut, explicit ``next_sop`` hint,
blocking ``asks``, empty-cart override, re-entry guard, ``MAX_ITERATIONS``
cap) are ported verbatim from v2's ``supervisor`` node.
"""

from __future__ import annotations

from typing import Any

from agno.agent import Agent

from agent_v3.models import chat_model
from agent_v3.sop_names import SOPName, SupervisorDecision
from agent_v3.state import last_user_message, load_cart

MAX_ITERATIONS = 4
HISTORY_TURNS = 8

# Step names used by the workflow's Router (must match the Step names in
# workflow.py). ``finalize`` ends the supervisor loop.
FINALIZE_STEP = "finalize"
DONE_SENTINEL = "__supervisor_done__"  # finalize step's StepOutput.content


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
    "hi", "hello", "hey", "yo", "sup", "thanks", "thank you", "ty", "ok",
    "okay", "cool", "nice", "lol", "bye", "goodbye", "see you",
}


def _is_likely_smalltalk(user_msg: str) -> bool:
    """Cheap heuristic to skip the LLM classifier for obvious greetings."""
    msg = user_msg.strip().lower().rstrip("!?.")
    if not msg or len(msg) > 40:
        return False
    if msg in _SMALLTALK_KEYWORDS:
        return True
    words = [w for w in msg.replace(",", " ").split() if w not in {"there", "you", "are", "how"}]
    return bool(words) and all(w in _SMALLTALK_KEYWORDS for w in words)


def _format_history(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role == "human":
            lines.append(f"USER: {content}")
        elif role == "ai" and content:
            lines.append(f"ASSISTANT: {content}")
    return "\n".join(lines) if lines else "(no prior turns)"


def _format_step_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "(none)"
    out: list[str] = []
    for r in results:
        line = f"- {r.get('sop')}: {r.get('summary', '')}"
        if r.get("asks"):
            line += f" asks={r['asks']}"
        if r.get("next_sop"):
            line += f" next_sop={r['next_sop']}"
        out.append(line)
    return "\n".join(out)


_CLASSIFIER: Agent | None = None


def _classifier_agent() -> Agent:
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = Agent(
            name="supervisor_classifier",
            model=chat_model(),
            instructions=_CLASSIFIER_PROMPT,
            output_schema=SupervisorDecision,
            telemetry=False,
        )
    return _CLASSIFIER


def classify_with_history(session_state: dict[str, Any]) -> SupervisorDecision:
    """Classify the next routing step from the shared session_state."""
    history = session_state.get("messages", [])[-HISTORY_TURNS:]
    user_payload = (
        f"Recent conversation:\n{_format_history(history)}\n\n"
        f"Cart step: {load_cart(session_state).step.value}\n"
        f"Step results this turn:\n{_format_step_results(session_state.get('step_results', []))}\n\n"
        "Decide done + next_sop."
    )
    resp = _classifier_agent().run(input=user_payload)
    parsed = resp.content
    if parsed is None or not isinstance(parsed, SupervisorDecision):
        return SupervisorDecision(done=False, next_sop=SOPName.PRODUCT_REC, reason="fallback")
    if not parsed.done and parsed.next_sop is None:
        parsed = parsed.model_copy(update={"next_sop": SOPName.PRODUCT_REC})
    return parsed


def supervisor_selector(step_input: Any, session_state: dict[str, Any]) -> str:
    """Agno Router selector: decide the next step name (or ``finalize``)."""
    iteration = session_state.get("iteration", 0)
    step_results = session_state.get("step_results", [])

    # 1. Hard cap: never loop forever.
    if iteration >= MAX_ITERATIONS:
        return FINALIZE_STEP

    # 2. Explicit next_sop hint from the previous step (compound-ask fast path).
    if step_results:
        last = step_results[-1]
        if last.get("next_sop"):
            session_state["active_sop"] = last["next_sop"]
            session_state["iteration"] = iteration + 1
            return last["next_sop"]
        # 3. Last SOP is waiting on the user (surfaced asks) — stop.
        if last.get("asks"):
            return FINALIZE_STEP

    # 4. Smalltalk shortcut: skip the classifier on obvious greetings.
    if not step_results and _is_likely_smalltalk(last_user_message(session_state)):
        return FINALIZE_STEP

    # 5. Classify with full context.
    decision = classify_with_history(session_state)
    if decision.done:
        return FINALIZE_STEP
    next_sop = decision.next_sop or SOPName.PRODUCT_REC

    # 5b. Empty-cart safety net (mirror of checkout_gate defense-in-depth).
    already_ran = {r.get("sop") for r in step_results}
    if (
        next_sop == SOPName.CHECKOUT
        and load_cart(session_state).step.value == "collecting_products"
        and SOPName.PRODUCT_REC.value not in already_ran
    ):
        next_sop = SOPName.PRODUCT_REC

    # 6. Don't re-enter a SOP we already ran this turn.
    if next_sop.value in already_ran:
        return FINALIZE_STEP

    session_state["active_sop"] = next_sop.value
    session_state["iteration"] = iteration + 1
    return next_sop.value
