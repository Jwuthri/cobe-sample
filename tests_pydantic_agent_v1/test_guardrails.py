"""Input guardrails — the pre-flight gate."""

from __future__ import annotations

from pydantic_agent_v1.runtime.guardrails import Blocklist, PiiRedact, run_input_guardrails


def test_no_rules_allows_everything():
    out = run_input_guardrails([], "hello there")
    assert out.allowed and out.text == "hello there" and not out.triggered


def test_blocklist_blocks_and_refuses():
    rules = [Blocklist(phrases=["hack the mainframe"], message="No.")]
    out = run_input_guardrails(rules, "please hack the mainframe")
    assert not out.allowed
    assert out.refusal == "No."
    assert out.triggered[0].type == "blocklist"


def test_pii_redacts_email_in_place():
    out = run_input_guardrails([PiiRedact()], "email me at a@b.com please")
    assert out.allowed
    assert "a@b.com" not in out.text
    assert "[redacted]" in out.text
    assert out.triggered[0].action == "redact"
