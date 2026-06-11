"""Guardrails — a self-contained input pre-flight engine (no LangChain).

agent_v4_1 leaned on LangChain's ``PIIMiddleware`` + middleware ``before_model``
hooks. This Agno port keeps the *same behaviour* with zero framework coupling:

  * the rules are plain objects with a ``run_input(text)`` step;
  * :func:`run_input_guardrails` threads the user text through every input rule
    BEFORE the team runs — a ``block`` short-circuits with an instant refusal, a
    ``redact`` rewrites the text so every downstream model sees only clean input.

Three rule types are registered: ``blocklist`` (phrase/regex), ``pii``
(regex-based redact/mask of common entities), and ``llm_judge`` (a small Agno
agent judges a natural-language policy; fails OPEN on error — demo availability
over strictness). The shopping demo configures none of these by default
(checkout is gated by the cart invariant, not a content filter); they are a
platform capability exercised by ``examples.EXAMPLE_AGENT_CONFIG`` + the tests.

Agno also ships native ``pre_hooks``/``BaseGuardrail`` support, but running the
gate as an explicit pre-flight step keeps the SSE ``{type:"guardrail"}`` contract
and the instant-refusal semantics identical to v4_1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from agent_agno_v1.core.config import GuardrailSpec
from agent_agno_v1.core.registry import GUARDRAILS

_DEFAULT_REFUSAL = "I'm not able to help with that request."

# Common PII patterns (intentionally small + conservative).
_PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"\b(?:\+?\d{1,2}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


# =============================================================================
# rule step result
# =============================================================================
@dataclass
class RuleStep:
    """One rule's verdict on the (possibly already-redacted) input text."""

    triggered: bool = False
    blocked: bool = False
    text: str | None = None  # rewritten text (for redact/mask)
    refusal: str | None = None


# =============================================================================
# rule implementations
# =============================================================================
class BlocklistRule:
    """Phrase / regex guardrail. Blocks (or redacts) on input."""

    type = "blocklist"

    def __init__(self, spec: GuardrailSpec) -> None:
        self.spec = spec
        self.phrases = tuple(p.lower() for p in (spec.params.get("phrases") or []))
        self.patterns = tuple(
            re.compile(p, re.IGNORECASE) for p in (spec.params.get("patterns") or [])
        )

    def _matches(self, text: str) -> bool:
        lowered = text.lower()
        if any(phrase in lowered for phrase in self.phrases):
            return True
        return any(pattern.search(text) for pattern in self.patterns)

    def run_input(self, text: str) -> RuleStep:
        if not self._matches(text):
            return RuleStep()
        if self.spec.action == "redact":
            new = text
            for pattern in self.patterns:
                new = pattern.sub("[redacted]", new)
            return RuleStep(triggered=True, text=new)
        return RuleStep(triggered=True, blocked=True, refusal=self.spec.message or _DEFAULT_REFUSAL)


class PiiRule:
    """Detect common PII entities; redact / mask / block on input."""

    type = "pii"

    def __init__(self, spec: GuardrailSpec) -> None:
        self.spec = spec
        entities = spec.params.get("entities") or [spec.params.get("entity", "email")]
        self.patterns = [(_PII_PATTERNS[e], e) for e in entities if e in _PII_PATTERNS]

    def run_input(self, text: str) -> RuleStep:
        hit = any(pattern.search(text) for pattern, _ in self.patterns)
        if not hit:
            return RuleStep()
        if self.spec.action == "block":
            return RuleStep(
                triggered=True, blocked=True, refusal=self.spec.message or _DEFAULT_REFUSAL
            )
        new = text
        for pattern, entity in self.patterns:
            replacement = "*" * 6 if self.spec.action == "mask" else f"[{entity}]"
            new = pattern.sub(replacement, new)
        return RuleStep(triggered=True, text=new)


class _PolicyVerdict:
    """Tiny structured-output schema for the llm_judge (avoids a hard pydantic dep here)."""


_POLICY_JUDGE_SYSTEM = """You are a content-policy checker.
Decide if the user message violates this policy:

<policy>
{policy}
</policy>

Judge by meaning, not keywords. Answer with a single word: VIOLATES or OK."""


class LlmJudgeRule:
    """Semantic guardrail: a small Agno agent judges the text against a policy.

    Fails OPEN on any error. ``params['judge']`` injects a ``Callable[[str], bool]``
    for tests (so no real model call is made). Otherwise a lazy Agno agent is built.
    """

    type = "llm_judge"

    def __init__(self, spec: GuardrailSpec) -> None:
        self.spec = spec
        self.policy = spec.params.get("policy")
        if not self.policy:
            raise ValueError("llm_judge guardrail requires params.policy")
        self.model_id = spec.params.get("model", "gpt-4.1-mini")
        self._injected: Callable[[str], bool] | None = spec.params.get("judge")
        self._agent: Any = None

    def _judge(self, text: str) -> bool:
        if self._injected is not None:
            return bool(self._injected(text))
        try:
            if self._agent is None:
                from agno.agent import Agent
                from agno.models.openai import OpenAIChat

                self._agent = Agent(
                    model=OpenAIChat(id=self.model_id, temperature=0.0),
                    system_message=_POLICY_JUDGE_SYSTEM.format(policy=self.policy),
                    telemetry=False,
                )
            out = self._agent.run(text)
            return "VIOLATES" in str(out.content or "").upper()
        except Exception:
            return False  # fail open

    def run_input(self, text: str) -> RuleStep:
        if not text.strip() or not self._judge(text):
            return RuleStep()
        return RuleStep(triggered=True, blocked=True, refusal=self.spec.message or _DEFAULT_REFUSAL)


# =============================================================================
# registry wiring
# =============================================================================
def register_builtin_guardrails() -> None:
    for name, factory in (
        ("blocklist", BlocklistRule),
        ("pii", PiiRule),
        ("llm_judge", LlmJudgeRule),
    ):
        if not GUARDRAILS.has(name):
            GUARDRAILS.register(name, factory, category="safety")


# =============================================================================
# pre-flight engine
# =============================================================================
@dataclass
class CompiledInputRule:
    spec: GuardrailSpec
    rule: Any  # has .run_input(text) -> RuleStep


def compile_input_rules(specs: list[GuardrailSpec]) -> list[CompiledInputRule]:
    """Compile only the input-side guardrails into runnable rules."""
    register_builtin_guardrails()
    return [
        CompiledInputRule(spec=g, rule=GUARDRAILS.get(g.type)(g)) for g in specs if g.on_input
    ]


@dataclass
class GuardrailHit:
    type: str
    action: str


@dataclass
class InputOutcome:
    allowed: bool
    text: str
    refusal: str | None = None
    triggered: list[GuardrailHit] = field(default_factory=list)


def run_input_guardrails(rules: list[CompiledInputRule], text: str) -> InputOutcome:
    """Run input guardrails before the team runs.

    Threads ``text`` through each rule: a redact rule rewrites it, a block rule
    short-circuits with a refusal.
    """
    current = text
    triggered: list[GuardrailHit] = []
    for compiled in rules:
        step = compiled.rule.run_input(current)
        if not step.triggered:
            continue
        triggered.append(GuardrailHit(compiled.spec.type, compiled.spec.action))
        if step.blocked:
            return InputOutcome(
                False, current, step.refusal or _DEFAULT_REFUSAL, triggered
            )
        if step.text is not None:
            current = step.text
    return InputOutcome(True, current, None, triggered)


__all__ = [
    "BlocklistRule",
    "PiiRule",
    "LlmJudgeRule",
    "CompiledInputRule",
    "GuardrailHit",
    "InputOutcome",
    "compile_input_rules",
    "run_input_guardrails",
    "register_builtin_guardrails",
]
