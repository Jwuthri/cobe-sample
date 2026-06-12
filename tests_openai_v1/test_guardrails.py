"""The input pre-flight guardrail engine (block / redact / judge)."""

from __future__ import annotations

from openai_agent_v1.core.config import GuardrailSpec
from openai_agent_v1.core.guardrails import compile_input_rules, run_input_guardrails


def _rules(*specs: GuardrailSpec):
    return compile_input_rules(list(specs))


def test_blocklist_blocks_and_refuses():
    rules = _rules(
        GuardrailSpec(
            type="blocklist",
            action="block",
            message="No legal advice.",
            params={"phrases": ["lawsuit"]},
        )
    )
    out = run_input_guardrails(rules, "I want to file a lawsuit")
    assert out.allowed is False
    assert out.refusal == "No legal advice."
    assert out.triggered[0].type == "blocklist"


def test_blocklist_allows_clean_text():
    rules = _rules(GuardrailSpec(type="blocklist", params={"phrases": ["lawsuit"]}))
    out = run_input_guardrails(rules, "show me hoodies")
    assert out.allowed is True
    assert out.text == "show me hoodies"
    assert out.triggered == []


def test_pii_redacts_email():
    rules = _rules(GuardrailSpec(type="pii", action="redact", params={"entity": "email"}))
    out = run_input_guardrails(rules, "email me at ada@example.com please")
    assert out.allowed is True
    assert "ada@example.com" not in out.text
    assert "[REDACTED_EMAIL]" in out.text
    assert out.triggered[0].type == "pii"


def test_llm_judge_blocks_with_injected_judge():
    rules = _rules(
        GuardrailSpec(
            type="llm_judge",
            action="block",
            message="Off topic.",
            params={"policy": "no politics", "judge": lambda text: "politics" in text.lower()},
        )
    )
    blocked = run_input_guardrails(rules, "let's talk POLITICS")
    assert blocked.allowed is False and blocked.refusal == "Off topic."
    ok = run_input_guardrails(rules, "show me hoodies")
    assert ok.allowed is True


def test_llm_judge_fails_open_on_error():
    def _boom(_text):
        raise RuntimeError("judge down")

    rules = _rules(GuardrailSpec(type="llm_judge", params={"policy": "x", "judge": _boom}))
    out = run_input_guardrails(rules, "anything")
    assert out.allowed is True  # judge error → fail open


def test_redact_then_block_chain():
    rules = _rules(
        GuardrailSpec(type="pii", action="redact", params={"entity": "email"}),
        GuardrailSpec(type="blocklist", action="block", message="nope", params={"phrases": ["bomb"]}),
    )
    out = run_input_guardrails(rules, "mail x@y.com about a bomb")
    assert out.allowed is False  # blocked by the second rule
    assert out.refusal == "nope"
