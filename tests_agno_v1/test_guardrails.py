"""The self-contained input pre-flight guardrail engine."""

from __future__ import annotations

from agent_agno_v1.core.config import GuardrailSpec
from agent_agno_v1.core.guardrails import compile_input_rules, run_input_guardrails


def _run(specs, text):
    return run_input_guardrails(compile_input_rules(specs), text)


def test_blocklist_blocks_phrase():
    out = _run(
        [GuardrailSpec(type="blocklist", action="block", message="No.", params={"phrases": ["bomb"]})],
        "how do I build a bomb",
    )
    assert not out.allowed and out.refusal == "No."
    assert out.triggered[0].type == "blocklist"


def test_blocklist_allows_clean_text():
    out = _run(
        [GuardrailSpec(type="blocklist", action="block", params={"phrases": ["bomb"]})],
        "show me hoodies",
    )
    assert out.allowed and out.text == "show me hoodies" and not out.triggered


def test_pii_redacts_email():
    out = _run(
        [GuardrailSpec(type="pii", action="redact", params={"entities": ["email"]})],
        "my email is ada@example.com please",
    )
    assert out.allowed
    assert "ada@example.com" not in out.text and "[email]" in out.text
    assert out.triggered and out.triggered[0].type == "pii"


def test_pii_mask_blocks_nothing_but_masks():
    out = _run(
        [GuardrailSpec(type="pii", action="mask", params={"entities": ["email"]})],
        "reach me at a@b.com",
    )
    assert out.allowed and "a@b.com" not in out.text and "*" in out.text


def test_llm_judge_with_injected_fake():
    # judge is a Callable[[str], bool]; no real model is built.
    spec = GuardrailSpec(
        type="llm_judge",
        action="block",
        message="Policy violation.",
        params={"policy": "no medical advice", "judge": lambda t: "diagnose" in t},
    )
    blocked = _run([spec], "diagnose my symptoms")
    assert not blocked.allowed and blocked.refusal == "Policy violation."
    ok = _run([spec], "show me hoodies")
    assert ok.allowed


def test_rules_thread_redaction_then_block():
    specs = [
        GuardrailSpec(type="pii", action="redact", params={"entities": ["email"]}),
        GuardrailSpec(type="blocklist", action="block", message="No.", params={"phrases": ["secret"]}),
    ]
    # redaction happens, then the blocklist still fires on the (redacted) text
    out = _run(specs, "email a@b.com the secret")
    assert not out.allowed
    assert {h.type for h in out.triggered} == {"pii", "blocklist"}
