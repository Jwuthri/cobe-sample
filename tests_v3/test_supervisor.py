"""Supervisor selector — pure routing logic (no LLM).

Ports tests/test_supervisor_loop.py to the Agno Router selector: instead of
a graph node returning ``Command(goto=...)``, ``supervisor_selector`` reads
the shared ``session_state`` and returns the next step name (or ``finalize``).
"""

from __future__ import annotations

from unittest.mock import patch

from agent_v3.checkout import CartService
from agent_v3.sop_names import SOPName, SupervisorDecision
from agent_v3.state import fresh_state, save_cart
from agent_v3.step_result import StepResult
from agent_v3.supervisor import (
    FINALIZE_STEP,
    MAX_ITERATIONS,
    _is_likely_smalltalk,
    supervisor_selector,
)

CLASSIFY = "agent_v3.supervisor.classify_with_history"


def _ss(messages=None, step_results=None, iteration=0, cart=None):
    s = fresh_state(user_id="u", session_id="s")
    if messages is not None:
        s["messages"] = messages
    if step_results is not None:
        s["step_results"] = [sr.model_dump(mode="json") for sr in step_results]
    s["iteration"] = iteration
    if cart is not None:
        save_cart(s, cart)
    return s


def test_loop_cap_routes_to_finalize():
    s = _ss(iteration=MAX_ITERATIONS)
    assert supervisor_selector(None, s) == FINALIZE_STEP


def test_next_sop_hint_followed_without_classifier():
    sr = StepResult(sop=SOPName.PRODUCT_REC, summary="added P-4", next_sop=SOPName.CHECKOUT)
    s = _ss(step_results=[sr], iteration=1)
    with patch(CLASSIFY) as classifier:
        classifier.side_effect = AssertionError("classifier should NOT be invoked")
        target = supervisor_selector(None, s)
    assert target == SOPName.CHECKOUT.value
    assert s["active_sop"] == SOPName.CHECKOUT.value
    assert s["iteration"] == 2


def test_classifier_done_routes_to_finalize():
    s = _ss(iteration=1)
    with patch(CLASSIFY) as classifier:
        classifier.return_value = SupervisorDecision(done=True, reason="all set")
        assert supervisor_selector(None, s) == FINALIZE_STEP


def test_classifier_picks_a_sop():
    s = _ss(iteration=0)
    with patch(CLASSIFY) as classifier:
        classifier.return_value = SupervisorDecision(
            done=False, next_sop=SOPName.PRODUCT_REC, reason="browsing"
        )
        target = supervisor_selector(None, s)
    assert target == SOPName.PRODUCT_REC.value
    assert s["active_sop"] == SOPName.PRODUCT_REC.value
    assert s["iteration"] == 1


def test_smalltalk_short_circuits_without_classifier():
    s = _ss(messages=[{"role": "human", "content": "hello"}])
    with patch(CLASSIFY) as classifier:
        classifier.side_effect = AssertionError("classifier must not run on smalltalk")
        assert supervisor_selector(None, s) == FINALIZE_STEP


def test_smalltalk_detection_is_conservative():
    assert _is_likely_smalltalk("hello")
    assert _is_likely_smalltalk("Hi there!")
    assert _is_likely_smalltalk("ok cool")
    assert not _is_likely_smalltalk("buy P-1")
    assert not _is_likely_smalltalk("where is my order ORD-7")
    assert not _is_likely_smalltalk("")
    assert not _is_likely_smalltalk(
        "hello, i'd like to know if you ship black hoodies to san francisco"
    )


def test_last_step_asks_short_circuits():
    sr = StepResult(sop=SOPName.CHECKOUT, summary="captured identity", asks=["street", "zip"])
    s = _ss(step_results=[sr], iteration=1)
    with patch(CLASSIFY) as classifier:
        classifier.side_effect = AssertionError("should not run when last has asks")
        assert supervisor_selector(None, s) == FINALIZE_STEP


def test_no_reentry_same_sop():
    sr = StepResult(sop=SOPName.PRODUCT_REC, summary="showed 3 results")
    s = _ss(step_results=[sr], iteration=1)
    with patch(CLASSIFY) as classifier:
        classifier.return_value = SupervisorDecision(done=False, next_sop=SOPName.PRODUCT_REC)
        assert supervisor_selector(None, s) == FINALIZE_STEP


def test_empty_cart_overrides_checkout_to_product_rec():
    s = _ss(messages=[{"role": "human", "content": "add the cap to my cart"}])
    with patch(CLASSIFY) as classifier:
        classifier.return_value = SupervisorDecision(done=False, next_sop=SOPName.CHECKOUT)
        target = supervisor_selector(None, s)
    assert target == SOPName.PRODUCT_REC.value
    assert s["active_sop"] == SOPName.PRODUCT_REC.value


def test_non_empty_cart_lets_checkout_through():
    svc = CartService()
    svc.add_item("P-1")
    s = _ss(cart=svc.cart)
    with patch(CLASSIFY) as classifier:
        classifier.return_value = SupervisorDecision(done=False, next_sop=SOPName.CHECKOUT)
        assert supervisor_selector(None, s) == SOPName.CHECKOUT.value


def test_compound_ask_two_iterations():
    s = _ss(iteration=0)
    with patch(CLASSIFY) as classifier:
        classifier.return_value = SupervisorDecision(done=False, next_sop=SOPName.PRODUCT_REC)
        assert supervisor_selector(None, s) == SOPName.PRODUCT_REC.value
    # simulate product_rec finishing with a checkout hand-off
    sr = StepResult(sop=SOPName.PRODUCT_REC, summary="added P-4", next_sop=SOPName.CHECKOUT)
    s["step_results"] = [sr.model_dump(mode="json")]
    assert supervisor_selector(None, s) == SOPName.CHECKOUT.value
