"""Input guardrail tests — deterministic, no model.

The guardrails are pre-flight: a refusal short-circuits the turn before any model
call, and a redaction rewrites the user text before it enters the transcript.
"""

from __future__ import annotations

from agent_openai_sdk_v1.runtime.guardrails import Blocklist, PiiRedact, run_input_guardrails


def test_blocklist_short_circuits_the_turn():
    rule = Blocklist(phrases=["forbidden"], message="I can't help with that.")
    outcome = run_input_guardrails([rule], "this is forbidden")
    assert not outcome.allowed
    assert outcome.refusal == "I can't help with that."
    assert outcome.triggered[0].type == "blocklist"
    assert outcome.triggered[0].action == "block"


def test_pii_redaction_rewrites_text_without_blocking():
    outcome = run_input_guardrails([PiiRedact()], "email me at user@example.com please")
    assert outcome.allowed
    assert "user@example.com" not in outcome.text
    assert "[redacted]" in outcome.text
    assert outcome.triggered[0].type == "pii"
    assert outcome.triggered[0].action == "redact"


def test_no_rules_no_changes():
    outcome = run_input_guardrails([], "hello there")
    assert outcome.allowed
    assert outcome.text == "hello there"
    assert outcome.triggered == []


def test_rules_run_in_order():
    """A redact rule and then a blocklist rule — the redact rewrites first, then
    the blocklist checks the rewritten text."""
    redact = PiiRedact()
    block = Blocklist(phrases=["[redacted]"], message="No PII allowed.")
    outcome = run_input_guardrails([redact, block], "contact user@example.com")
    assert not outcome.allowed
    assert outcome.refusal == "No PII allowed."
    # both guardrails recorded
    actions = [(h.type, h.action) for h in outcome.triggered]
    assert ("pii", "redact") in actions
    assert ("blocklist", "block") in actions
