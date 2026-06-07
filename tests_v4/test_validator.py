"""The validator is now a minimal structural net (no regex, no gate).

All content checks were removed — the writer (an LLM with the full cart
state) owns content, including the no-false-confirmation rule. The only
thing left is: retry once if the writer produced no text.
"""

from __future__ import annotations

from agent_v4.graph import validator
from agent_v4.state import MAX_VALIDATOR_RETRIES, AgentState
from langgraph.types import Command


def _state(**kw) -> AgentState:
    return AgentState(user_id="u", session_id="s", **kw)


def test_validator_emits_clean_draft():
    cmd = validator(_state(draft_response="Here you go."))
    assert isinstance(cmd, Command)
    assert cmd.goto == "emit"


def test_validator_no_longer_rejects_placeholders_or_unsafe_words():
    # The placeholder / unsafe / length regexes were removed on purpose —
    # these now pass straight through to emit.
    cmd = validator(_state(draft_response="Hi {{name}}, you stupid customer."))
    assert cmd.goto == "emit"


def test_validator_retries_writer_on_empty_draft():
    cmd = validator(_state(draft_response="", response_attempts=0))
    assert cmd.goto == "writer"
    assert cmd.update["response_attempts"] == 1


def test_validator_falls_back_after_max_retries_on_empty():
    cmd = validator(_state(draft_response="", response_attempts=MAX_VALIDATOR_RETRIES))
    assert cmd.goto == "emit"
    assert "rephrase" in cmd.update["draft_response"].lower()
