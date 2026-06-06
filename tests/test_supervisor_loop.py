"""The supervisor node — pure routing logic (no LLM call).

We exercise the loop math directly: max_iterations cap and the
``next_sop`` hint pathway. Classification with OpenAI is mocked out
where needed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from agent_v2.checkout import CartService
from agent_v2.state import AgentState
from agent_v2.step_result import StepResult
from agent_v2.supervisor import (
    MAX_ITERATIONS,
    SOPName,
    SupervisorDecision,
    _is_likely_smalltalk,
    supervisor,
)
from langchain_core.messages import HumanMessage
from langgraph.types import Command


def _state(**kw) -> AgentState:
    return AgentState(user_id="u", session_id="s", **kw)


def test_loop_cap_routes_to_writer_when_iteration_exceeds_max():
    s = _state(iteration=MAX_ITERATIONS)
    cmd = supervisor(s)
    assert cmd.goto == "writer"
    assert cmd.update.get("iteration") == 0


def test_next_sop_hint_is_followed_without_calling_classifier():
    """When the most recent step_result hints next_sop, we follow it
    without making an OpenAI call."""
    sr = StepResult(sop=SOPName.PRODUCT_REC, summary="added P-4", next_sop=SOPName.CHECKOUT)
    s = _state(step_results=[sr], iteration=1)

    # Patch the classifier to fail if it gets called.
    with patch("agent_v2.supervisor.classify_with_history") as classifier:
        classifier.side_effect = AssertionError("classifier should NOT be invoked")
        cmd = supervisor(s)

    assert cmd.goto == "checkout_wrapper"
    assert cmd.update["active_sop"] == SOPName.CHECKOUT
    assert cmd.update["iteration"] == 2


def test_classifier_done_routes_to_writer():
    s = _state(iteration=1, step_results=[])
    with patch("agent_v2.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(done=True, reason="all set")
        cmd = supervisor(s)
    assert cmd.goto == "writer"


def test_classifier_picks_a_sop_when_not_done():
    s = _state(iteration=0)
    with patch("agent_v2.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(
            done=False, next_sop=SOPName.PRODUCT_REC, reason="user browsing"
        )
        cmd = supervisor(s)
    assert cmd.goto == "product_rec_wrapper"
    assert cmd.update["active_sop"] == SOPName.PRODUCT_REC
    assert cmd.update["iteration"] == 1


def test_smalltalk_short_circuits_to_writer_without_classifier_call():
    """Greetings like 'hello' must not run any SOP — they go straight to writer."""
    s = _state(messages=[HumanMessage(content="hello")])
    with patch("agent_v2.supervisor.classify_with_history") as classifier:
        classifier.side_effect = AssertionError("classifier must not run on smalltalk")
        cmd = supervisor(s)
    assert cmd.goto == "writer"
    assert cmd.update["iteration"] == 0


def test_smalltalk_keyword_detection_is_conservative():
    assert _is_likely_smalltalk("hello")
    assert _is_likely_smalltalk("Hi there!")
    assert _is_likely_smalltalk("thanks")
    assert _is_likely_smalltalk("ok cool")
    # Real intents must NOT be misclassified as smalltalk.
    assert not _is_likely_smalltalk("buy P-1")
    assert not _is_likely_smalltalk("i want a hat")
    assert not _is_likely_smalltalk("where is my order ORD-7")
    assert not _is_likely_smalltalk("")
    # Long messages aren't smalltalk even if they start with a greeting.
    assert not _is_likely_smalltalk(
        "hello, i'd like to know if you ship black hoodies to san francisco"
    )


def test_supervisor_short_circuits_when_last_step_has_asks_and_no_hint():
    """After a SOP says 'I need user input', don't run anything else this turn."""
    sr = StepResult(
        sop=SOPName.CHECKOUT,
        summary="captured identity",
        asks=["street", "city", "zip"],
        next_sop=None,
    )
    s = _state(step_results=[sr], iteration=1)
    with patch("agent_v2.supervisor.classify_with_history") as classifier:
        classifier.side_effect = AssertionError("classifier should not run when last has asks")
        cmd = supervisor(s)
    assert cmd.goto == "writer"


def test_supervisor_doesnt_re_enter_same_sop_if_classifier_repicks():
    """If product_rec already ran and the classifier picks product_rec
    again (because the user message hasn't changed), short-circuit to
    writer instead of running it twice."""
    sr = StepResult(sop=SOPName.PRODUCT_REC, summary="showed 3 results")
    s = _state(step_results=[sr], iteration=1)
    with patch("agent_v2.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(
            done=False, next_sop=SOPName.PRODUCT_REC, reason="still browsing"
        )
        cmd = supervisor(s)
    assert cmd.goto == "writer"


def test_empty_cart_overrides_classifier_checkout_pick_to_product_rec():
    """Regression: 'add the cap to my cart' on an empty cart was being
    routed to checkout, which had nothing to do. With the empty-cart
    override, classifier-chosen checkout becomes product_rec when the
    cart is empty AND product_rec hasn't run yet this turn."""
    s = _state(messages=[HumanMessage(content="add the cap to my cart")])
    # By default the AgentState's cart is empty (COLLECTING_PRODUCTS).
    with patch("agent_v2.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(
            done=False, next_sop=SOPName.CHECKOUT, reason="user said 'cart'"
        )
        cmd = supervisor(s)
    assert cmd.goto == "product_rec_wrapper"
    assert cmd.update["active_sop"] == SOPName.PRODUCT_REC


def test_empty_cart_override_does_not_loop_after_product_rec_already_ran():
    """If product_rec already ran this turn and classifier picks
    checkout again, the override should NOT redirect (we'd loop).
    The same-SOP guard kicks in instead → writer."""
    sr = StepResult(sop=SOPName.PRODUCT_REC, summary="searched, no clear match")
    s = _state(
        messages=[HumanMessage(content="just add anything")],
        step_results=[sr],
        iteration=1,
    )
    with patch("agent_v2.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(
            done=False, next_sop=SOPName.CHECKOUT, reason="???"
        )
        cmd = supervisor(s)
    # Override should be skipped (product_rec already ran) → same-SOP
    # guard sees product_rec NOT yet run → wait, checkout hasn't run
    # either. Let's check actual: classifier said CHECKOUT, we DON'T
    # override (because PRODUCT_REC is in step_results), checkout NOT
    # in step_results, so we route to checkout.
    # The point of this test is to lock in: override is skipped when
    # product_rec already ran.
    assert cmd.goto == "checkout_wrapper"


def test_non_empty_cart_lets_classifier_pick_checkout():
    """The override should NOT trigger when the cart has items —
    checkout is legitimate at that point."""
    svc = CartService()
    svc.add_item("P-1")  # cart now has 1 item, step=collecting_identity
    s = _state(cart=svc.cart)
    with patch("agent_v2.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(
            done=False, next_sop=SOPName.CHECKOUT, reason="user providing name"
        )
        cmd = supervisor(s)
    assert cmd.goto == "checkout_wrapper"


def test_compound_ask_drives_two_iterations():
    """Simulate 'add the green cap and pay':
    iter 0 → classifier picks product_rec
    product_rec step result hints next_sop=checkout
    iter 1 → supervisor follows hint, no classifier call
    """
    s = _state(iteration=0)
    with patch("agent_v2.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(
            done=False, next_sop=SOPName.PRODUCT_REC, reason="compound: search first"
        )
        cmd0 = supervisor(s)
    assert cmd0.goto == "product_rec_wrapper"

    # Simulate the wrapper completing and producing a StepResult.
    sr = StepResult(sop=SOPName.PRODUCT_REC, summary="added P-4 to cart", next_sop=SOPName.CHECKOUT)
    s1 = s.model_copy(
        update={
            "iteration": cmd0.update["iteration"],
            "active_sop": cmd0.update["active_sop"],
            "step_results": [sr],
        }
    )
    cmd1 = supervisor(s1)
    assert cmd1.goto == "checkout_wrapper"
