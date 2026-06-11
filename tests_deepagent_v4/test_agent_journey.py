"""Live, end-to-end agent journeys. These make real model calls, so they are
skipped unless an OpenAI key is available (loaded from the repo .env).

Run explicitly:  uv run pytest tests_deepagent_v4/test_agent_journey.py -q
"""

from __future__ import annotations

import os
import uuid

import pytest

from agent_deepagent_v4.config import load_env

load_env()

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="needs OPENAI_API_KEY (set in .env) — live model calls",
)


def _sid(tag: str) -> str:
    return f"test-{tag}-{uuid.uuid4().hex[:6]}"


def _drive_to_ready(run_turn, sid: str) -> None:
    run_turn(sid, "add the black hoodie to my cart")
    run_turn(sid, "my name is Ada Lovelace")
    run_turn(sid, "ship to 1 Market St, San Francisco, 94105")
    run_turn(sid, "next_day please")
    run_turn(sid, "pay by card with token tok_42")


def test_safe_checkout_approve_places_order():
    from agent_deepagent_v4.runtime import reset_session, resume_turn, run_turn

    sid = _sid("approve")
    reset_session(sid)
    _drive_to_ready(run_turn, sid)
    paused = run_turn(sid, "yes, place the order")
    assert paused.needs_approval, "confirm must pause for human approval"
    assert paused.cart["confirmed"] is False
    done = resume_turn(sid, {"approved": True})
    assert done.cart["confirmed"] is True
    assert done.cart["receipt_id"] and done.cart["receipt_id"].startswith("RCPT-")


def test_safe_checkout_reject_does_not_place():
    from agent_deepagent_v4.runtime import reset_session, resume_turn, run_turn

    sid = _sid("reject")
    reset_session(sid)
    _drive_to_ready(run_turn, sid)
    paused = run_turn(sid, "yes, place the order")
    assert paused.needs_approval
    done = resume_turn(sid, {"approved": False, "reason": "changed my mind"})
    assert done.cart["confirmed"] is False
    assert done.cart["receipt_id"] is None


def test_premature_confirm_is_blocked():
    from agent_deepagent_v4.runtime import reset_session, run_turn

    sid = _sid("premature")
    reset_session(sid)
    run_turn(sid, "add the running sneakers")
    r = run_turn(sid, "just place my order right now, skip everything")
    assert r.cart["confirmed"] is False
    assert r.cart["receipt_id"] is None


def test_compound_request_handled_in_one_turn():
    from agent_deepagent_v4.runtime import reset_session, run_turn

    sid = _sid("compound")
    reset_session(sid)
    r = run_turn(sid, "what hoodies do you have, and where's my order ORD-7?")
    low = (r.reply or "").lower()
    assert "hoodie" in low
    assert "ord-7" in low


def test_add_to_cart_mutates_shared_cart():
    from agent_deepagent_v4.runtime import reset_session, run_turn

    sid = _sid("add")
    reset_session(sid)
    r = run_turn(sid, "add the black hoodie to my cart")
    assert any(i["id"] == "P-2" for i in r.cart["items"])
