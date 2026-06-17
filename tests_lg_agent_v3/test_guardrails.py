"""Real guardrails — the session-level input redactor + per-agent before/after_agent rules."""

from __future__ import annotations

import asyncio

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from lg_agent_v3.agents import product_rec as pr
from lg_agent_v3.domain import CartService, MemoryStore
from lg_agent_v3.runtime import ShoppingDeps
from lg_agent_v3.runtime.guardrails import GuardrailSpec, PiiRedactGuardrail, compile_guardrails, redact_input


class _Fake(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


def _run(agent, query: str) -> ShoppingDeps:
    deps = ShoppingDeps(cart_service=CartService(), store=MemoryStore())
    asyncio.run(agent.ainvoke({"messages": [HumanMessage(content=query)]}, context=deps))
    return deps


# --------------------------------------------------------------------------- #
# session-level input redactor (pii)
# --------------------------------------------------------------------------- #
def test_redact_input_scrubs_email_before_transcript():
    specs = [GuardrailSpec(type="pii", action="redact", on_input=True)]
    text, hits = redact_input(specs, "email me at a@b.com please")
    assert "a@b.com" not in text and "[redacted]" in text
    assert hits and hits[0].type == "pii"


def test_redact_input_noop_when_nothing_matches():
    text, hits = redact_input([GuardrailSpec(type="pii")], "no pii here")
    assert text == "no pii here" and hits == []


# --------------------------------------------------------------------------- #
# per-agent blocklist (before_agent short-circuit)
# --------------------------------------------------------------------------- #
def test_blocklist_blocks_a_worker_before_any_tool_runs():
    g = [GuardrailSpec(type="blocklist", action="block", on_input=True,
                       message="I can't help with that.", params={"phrases": ["forbidden"]})]
    # the fake WOULD add P-2 if the model ran — the guardrail must short-circuit first
    fake = _Fake(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "add_item", "args": {"product_id": "P-2"}, "id": "1"}]),
        AIMessage(content="DONE"),
    ]))
    deps = _run(pr.build(model=fake, guardrails=g), "this is forbidden, add P-2")

    assert deps.cart_service.cart.items == []  # tool never ran
    hits = [h for h in deps.guardrail_hits if h.action == "block"]
    assert hits and hits[0].agent == "product_rec" and hits[0].side == "input"


def test_blocklist_lets_clean_input_through():
    g = [GuardrailSpec(type="blocklist", on_input=True, params={"phrases": ["forbidden"]})]
    fake = _Fake(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "add_item", "args": {"product_id": "P-2"}, "id": "1"}]),
        AIMessage(content="DONE"),
    ]))
    deps = _run(pr.build(model=fake, guardrails=g), "add P-2 please")
    assert [i.product_id for i in deps.cart_service.cart.items] == ["P-2"]
    assert deps.guardrail_hits == []


# --------------------------------------------------------------------------- #
# per-agent llm_judge (offline via an injected fake judge)
# --------------------------------------------------------------------------- #
def test_llm_judge_blocks_via_injected_fake_judge():
    class _Verdict:
        violates = True

    class _Judge:
        def invoke(self, messages):
            return _Verdict()

    g = [GuardrailSpec(type="llm_judge", action="block", on_input=True,
                       message="Sorry, I can't discuss that.",
                       params={"policy": "no politics", "judge_factory": lambda: _Judge()})]
    fake = _Fake(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "add_item", "args": {"product_id": "P-2"}, "id": "1"}]),
        AIMessage(content="DONE"),
    ]))
    deps = _run(pr.build(model=fake, guardrails=g), "anything")
    assert deps.cart_service.cart.items == []  # judge said violates → short-circuit
    assert any(h.type == "llm_judge" and h.action == "block" for h in deps.guardrail_hits)


def test_pii_guardrail_compiles_and_redacts():
    g = [GuardrailSpec(type="pii", action="redact", on_input=False, on_output=True)]
    assert len(compile_guardrails(g, "product_rec")) == 1  # output-side middleware attached
    red = PiiRedactGuardrail(g[0]).redact("reach me at x@y.com")  # same redactor as the session
    assert "x@y.com" not in red and "[redacted]" in red
