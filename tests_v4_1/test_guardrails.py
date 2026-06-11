"""Guardrails: input short-circuit without a model call, redaction, output replace."""

from __future__ import annotations

from types import SimpleNamespace

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agent_v4_1.core.config import GuardrailSpec
from agent_v4_1.core.guardrails import (
    BlocklistGuardrail,
    compile_input_rules,
    run_input_guardrails,
)


class _ExplodingModel(GenericFakeChatModel):
    def _generate(self, *a, **k):
        raise AssertionError("model was called — guardrail did not short-circuit")

    def bind_tools(self, tools, **kw):
        return self


def test_blocklist_input_short_circuits_without_model_call():
    agent = create_agent(
        model=_ExplodingModel(messages=iter([AIMessage(content="x")])),
        tools=[],
        system_prompt="hi",
        middleware=[BlocklistGuardrail(phrases=["sue"], message="No legal advice.")],
    )
    out = agent.invoke({"messages": [HumanMessage(content="I want to sue them")]})
    assert out["messages"][-1].content == "No legal advice."  # model never ran


def test_run_input_guardrails_blocks_on_phrase():
    rules = compile_input_rules(
        [GuardrailSpec(type="blocklist", message="nope", params={"phrases": ["sue"]})]
    )
    outcome = run_input_guardrails(rules, "can I sue?")
    assert outcome.allowed is False
    assert outcome.refusal == "nope"
    assert outcome.triggered[0].type == "blocklist"


def test_run_input_guardrails_allows_clean():
    rules = compile_input_rules(
        [GuardrailSpec(type="blocklist", message="nope", params={"phrases": ["sue"]})]
    )
    outcome = run_input_guardrails(rules, "where is my order?")
    assert outcome.allowed is True
    assert outcome.text == "where is my order?"
    assert outcome.triggered == []


def test_pii_redact_rewrites_input_text():
    rules = compile_input_rules(
        [GuardrailSpec(type="pii", action="redact", params={"entity": "email"})]
    )
    outcome = run_input_guardrails(rules, "email me at alice@example.com please")
    assert outcome.allowed is True
    assert "alice@example.com" not in outcome.text
    assert "REDACTED" in outcome.text.upper()


def test_blocklist_output_replaces_by_id():
    g = BlocklistGuardrail(phrases=["secret"], message="[blocked]", on_input=False, on_output=True)
    ai = AIMessage(content="here is the secret value", id="ai-1")
    result = g.after_model({"messages": [HumanMessage(content="q"), ai]}, None)
    assert result is not None
    replacement = result["messages"][0]
    assert replacement.content == "[blocked]"
    assert replacement.id == "ai-1"  # same id → reducer replaces, no leftover text


def _judge_factory(violates: bool):
    class _J:
        def invoke(self, messages):
            return SimpleNamespace(violates=violates, reason="")

    return lambda: _J()


def test_llm_judge_blocks_on_violation():
    rules = compile_input_rules(
        [
            GuardrailSpec(
                type="llm_judge",
                message="off topic",
                params={"policy": "no politics", "judge_factory": _judge_factory(True)},
            )
        ]
    )
    outcome = run_input_guardrails(rules, "tell me about the election")
    assert outcome.allowed is False
    assert outcome.refusal == "off topic"


def test_llm_judge_allows_when_not_violating():
    rules = compile_input_rules(
        [
            GuardrailSpec(
                type="llm_judge",
                params={"policy": "no politics", "judge_factory": _judge_factory(False)},
            )
        ]
    )
    assert run_input_guardrails(rules, "where's my order").allowed is True


def test_llm_judge_fails_open_on_error():
    def _boom():
        class _J:
            def invoke(self, messages):
                raise RuntimeError("judge down")

        return _J()

    rules = compile_input_rules(
        [GuardrailSpec(type="llm_judge", params={"policy": "x", "judge_factory": _boom})]
    )
    # judge error → fail open (allowed)
    assert run_input_guardrails(rules, "anything").allowed is True
