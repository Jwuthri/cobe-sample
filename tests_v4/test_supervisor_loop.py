"""The supervisor node — pure routing logic (no LLM call).

We exercise the loop math directly: max_iterations cap, the ``next_sop``
hint pathway, smalltalk short-circuit, empty-cart override, and the
same-leaf re-entry guard. Classification with OpenAI is mocked out.
"""

from __future__ import annotations

from unittest.mock import patch

from agent_v4 import ids
from agent_v4.checkout import CartService
from agent_v4.state import AgentState
from agent_v4.step_result import StepResult
from agent_v4.supervisor import (
    MAX_ITERATIONS,
    SupervisorDecision,
    _is_likely_smalltalk,
    supervisor,
)
from langchain_core.messages import HumanMessage


def _state(**kw) -> AgentState:
    return AgentState(user_id="u", session_id="s", **kw)


def test_loop_cap_routes_to_writer_when_iteration_exceeds_max():
    s = _state(iteration=MAX_ITERATIONS)
    cmd = supervisor(s)
    assert cmd.goto == "writer"
    assert cmd.update.get("iteration") == 0


def test_next_sop_hint_is_followed_without_calling_classifier():
    sr = StepResult(sop=ids.PRODUCT_REC, summary="added P-4", next_sop=ids.CHECKOUT)
    s = _state(step_results=[sr], iteration=1)
    with patch("agent_v4.supervisor.classify_with_history") as classifier:
        classifier.side_effect = AssertionError("classifier should NOT be invoked")
        cmd = supervisor(s)
    assert cmd.goto == "checkout_wrapper"
    assert cmd.update["active_sop"] == ids.CHECKOUT
    assert cmd.update["iteration"] == 2


def test_classifier_done_routes_to_writer():
    s = _state(iteration=1, step_results=[])
    with patch("agent_v4.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(done=True, reason="all set")
        cmd = supervisor(s)
    assert cmd.goto == "writer"


def test_classifier_picks_a_leaf_when_not_done():
    s = _state(iteration=0)
    with patch("agent_v4.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(
            done=False, next_sop=ids.PRODUCT_REC, reason="user browsing"
        )
        cmd = supervisor(s)
    assert cmd.goto == "product_rec_wrapper"
    assert cmd.update["active_sop"] == ids.PRODUCT_REC
    assert cmd.update["iteration"] == 1


def test_smalltalk_short_circuits_to_writer_without_classifier_call():
    s = _state(messages=[HumanMessage(content="hello")])
    with patch("agent_v4.supervisor.classify_with_history") as classifier:
        classifier.side_effect = AssertionError("classifier must not run on smalltalk")
        cmd = supervisor(s)
    assert cmd.goto == "writer"
    assert cmd.update["iteration"] == 0


def test_smalltalk_keyword_detection_is_conservative():
    assert _is_likely_smalltalk("hello")
    assert _is_likely_smalltalk("Hi there!")
    assert _is_likely_smalltalk("thanks")
    assert _is_likely_smalltalk("ok cool")
    assert not _is_likely_smalltalk("buy P-1")
    assert not _is_likely_smalltalk("i want a hat")
    assert not _is_likely_smalltalk("where is my order ORD-7")
    assert not _is_likely_smalltalk("")
    assert not _is_likely_smalltalk(
        "hello, i'd like to know if you ship black hoodies to san francisco"
    )


def test_single_intent_asks_routes_to_writer_when_classifier_says_done():
    # A leaf asked the user a follow-up and there's no OTHER unhandled
    # intent -> the classifier (now consulted, no asks short-circuit)
    # returns done -> writer.
    sr = StepResult(
        sop=ids.CHECKOUT, summary="captured identity", asks=["street", "city", "zip"], next_sop=None
    )
    s = _state(step_results=[sr], iteration=1)
    with patch("agent_v4.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(done=True, reason="only intent handled")
        cmd = supervisor(s)
    assert cmd.goto == "writer"


def test_compound_ask_runs_second_leaf_after_first_returns_asks():
    # Regression: "what hoodies do you have, and where's my order ORD-7?"
    # product_rec handled the product part and asked the user to pick one;
    # the order-status part is still unhandled, so the supervisor must route
    # to order_status rather than bail to the writer (the old asks shortcut).
    sr = StepResult(sop=ids.PRODUCT_REC, summary="showed 1 product", asks=["pick a product id"])
    s = _state(step_results=[sr], iteration=1)
    with patch("agent_v4.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(
            done=False, next_sop=ids.ORDER_STATUS, reason="order part still unhandled"
        )
        cmd = supervisor(s)
    assert cmd.goto == "order_status_wrapper"
    assert cmd.update["active_sop"] == ids.ORDER_STATUS


def test_supervisor_doesnt_re_enter_same_leaf_if_classifier_repicks():
    sr = StepResult(sop=ids.PRODUCT_REC, summary="showed 3 results")
    s = _state(step_results=[sr], iteration=1)
    with patch("agent_v4.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(
            done=False, next_sop=ids.PRODUCT_REC, reason="still browsing"
        )
        cmd = supervisor(s)
    assert cmd.goto == "writer"


def test_empty_cart_overrides_classifier_checkout_pick_to_product_rec():
    s = _state(messages=[HumanMessage(content="add the cap to my cart")])
    with patch("agent_v4.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(
            done=False, next_sop=ids.CHECKOUT, reason="user said 'cart'"
        )
        cmd = supervisor(s)
    assert cmd.goto == "product_rec_wrapper"
    assert cmd.update["active_sop"] == ids.PRODUCT_REC


def test_empty_cart_override_does_not_loop_after_product_rec_already_ran():
    sr = StepResult(sop=ids.PRODUCT_REC, summary="searched, no clear match")
    s = _state(
        messages=[HumanMessage(content="just add anything")], step_results=[sr], iteration=1
    )
    with patch("agent_v4.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(
            done=False, next_sop=ids.CHECKOUT, reason="???"
        )
        cmd = supervisor(s)
    assert cmd.goto == "checkout_wrapper"


def test_non_empty_cart_lets_classifier_pick_checkout():
    svc = CartService()
    svc.add_item("P-1")
    s = _state(cart=svc.cart)
    with patch("agent_v4.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(
            done=False, next_sop=ids.CHECKOUT, reason="user providing name"
        )
        cmd = supervisor(s)
    assert cmd.goto == "checkout_wrapper"


def test_compound_ask_drives_two_iterations():
    s = _state(iteration=0)
    with patch("agent_v4.supervisor.classify_with_history") as classifier:
        classifier.return_value = SupervisorDecision(
            done=False, next_sop=ids.PRODUCT_REC, reason="compound: search first"
        )
        cmd0 = supervisor(s)
    assert cmd0.goto == "product_rec_wrapper"

    sr = StepResult(sop=ids.PRODUCT_REC, summary="added P-4 to cart", next_sop=ids.CHECKOUT)
    s1 = s.model_copy(
        update={
            "iteration": cmd0.update["iteration"],
            "active_sop": cmd0.update["active_sop"],
            "step_results": [sr],
        }
    )
    cmd1 = supervisor(s1)
    assert cmd1.goto == "checkout_wrapper"
