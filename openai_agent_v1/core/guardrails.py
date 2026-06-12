"""Guardrails — a framework-agnostic input pre-flight engine.

agent_v4_1 built guardrails on LangChain middleware; this clean-room port runs on
the OpenAI Agents SDK, so the same three rule types (``pii`` / ``blocklist`` /
``llm_judge``) are reimplemented as plain, dependency-light Python with an
identical *behavioural* surface to the session + tests:

  * :func:`compile_input_rules` + :func:`run_input_guardrails` — the pre-flight
    step run BEFORE any model call. A block short-circuits with an instant
    refusal; a redact rewrites the user text before it enters the transcript, so
    every downstream model sees only clean input.

The engine is the active path (it is what the shopping session wires); output-side
guardrails are available via :func:`compile_guardrails` for callers that want to
attach them to an SDK agent's ``output_guardrails``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from openai_agent_v1.core.config import GuardrailSpec
from openai_agent_v1.core.registry import GUARDRAILS

_DEFAULT_REFUSAL = "I'm not able to help with that request."


# =============================================================================
# result of one guardrail checking one piece of input
# =============================================================================
@dataclass
class InputCheck:
    triggered: bool = False
    blocked: bool = False
    refusal: str | None = None
    new_text: str | None = None  # set when the rule rewrote (redacted) the text


class Guardrail:
    """Base guardrail — implement :meth:`check_input` (and optionally output)."""

    def __init__(self, action: str = "block", message: str | None = None) -> None:
        self.action = action
        self.message = message or _DEFAULT_REFUSAL

    def check_input(self, text: str) -> InputCheck:  # pragma: no cover - overridden
        return InputCheck()


# =============================================================================
# blocklist
# =============================================================================
class BlocklistGuardrail(Guardrail):
    """Phrase / regex guardrail. Blocks (or flags) on a match."""

    def __init__(
        self,
        phrases: list[str] | None = None,
        patterns: list[str] | None = None,
        action: str = "block",
        message: str | None = None,
    ) -> None:
        super().__init__(action=action, message=message)
        self.phrases = tuple(p.lower() for p in (phrases or []))
        self.patterns = tuple(re.compile(p, re.IGNORECASE) for p in (patterns or []))

    def _matches(self, text: str) -> bool:
        lowered = text.lower()
        if any(phrase in lowered for phrase in self.phrases):
            return True
        return any(pattern.search(text) for pattern in self.patterns)

    def check_input(self, text: str) -> InputCheck:
        if not self._matches(text):
            return InputCheck()
        if self.action == "flag":
            return InputCheck(triggered=True)
        return InputCheck(triggered=True, blocked=True, refusal=self.message)


# =============================================================================
# pii (regex-based; entity → pattern)
# =============================================================================
_PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"\b(?:\+?\d{1,2}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "url": re.compile(r"https?://\S+"),
}


class PIIGuardrail(Guardrail):
    """Detect + redact / mask / block a single PII entity."""

    def __init__(self, entity: str = "email", action: str = "redact", message: str | None = None) -> None:
        super().__init__(action=action, message=message)
        self.entity = entity
        self.pattern = _PII_PATTERNS.get(entity, _PII_PATTERNS["email"])

    def _mask(self, value: str) -> str:
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]

    def check_input(self, text: str) -> InputCheck:
        if not self.pattern.search(text):
            return InputCheck()
        if self.action == "block":
            return InputCheck(triggered=True, blocked=True, refusal=self.message)
        if self.action == "mask":
            new = self.pattern.sub(lambda m: self._mask(m.group(0)), text)
        else:  # redact / hash / flag → redact placeholder
            new = self.pattern.sub(f"[REDACTED_{self.entity.upper()}]", text)
        return InputCheck(triggered=True, new_text=new)


# =============================================================================
# llm_judge
# =============================================================================
_POLICY_JUDGE_SYSTEM = """You are a content-policy checker.
Decide if the message violates this policy:

<policy>
{policy}
</policy>

Judge by meaning, not keywords. Answer with a single word: VIOLATES or OK."""


class LLMJudgeGuardrail(Guardrail):
    """Semantic guardrail: a small model judges the text against a policy.

    On a judge error it FAILS OPEN (does not block) — demo availability over
    strictness. ``judge`` (a ``Callable[[str], bool]``) is injectable for tests so
    no real model call is made.
    """

    def __init__(
        self,
        policy: str,
        model: str = "gpt-5.4-mini",
        action: str = "block",
        message: str | None = None,
        judge: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(action=action, message=message)
        self.policy = policy
        self.model = model
        self._judge = judge

    def _default_judge(self, text: str) -> bool:
        from openai import OpenAI  # local import; only on the real path

        client = OpenAI()
        resp = client.responses.create(
            model=self.model,
            instructions=_POLICY_JUDGE_SYSTEM.format(policy=self.policy),
            input=text,
        )
        verdict = (resp.output_text or "").strip().upper()
        return verdict.startswith("VIOLATES")

    def _violates(self, text: str) -> bool:
        if not text.strip():
            return False
        try:
            judge = self._judge or self._default_judge
            return bool(judge(text))
        except Exception:
            return False  # fail open

    def check_input(self, text: str) -> InputCheck:
        if self._violates(text):
            return InputCheck(triggered=True, blocked=True, refusal=self.message)
        return InputCheck()


# =============================================================================
# factories + registration
# =============================================================================
def _guardrail_pii(gr: GuardrailSpec) -> Guardrail:
    return PIIGuardrail(entity=gr.params.get("entity", "email"), action=gr.action, message=gr.message)


def _guardrail_blocklist(gr: GuardrailSpec) -> Guardrail:
    return BlocklistGuardrail(
        phrases=gr.params.get("phrases"),
        patterns=gr.params.get("patterns"),
        action=gr.action,
        message=gr.message,
    )


def _guardrail_llm_judge(gr: GuardrailSpec) -> Guardrail:
    policy = gr.params.get("policy")
    if not policy:
        raise ValueError("llm_judge guardrail requires params.policy")
    return LLMJudgeGuardrail(
        policy=policy,
        model=gr.params.get("model", "gpt-5.4-mini"),
        action=gr.action,
        message=gr.message,
        judge=gr.params.get("judge"),
    )


def register_builtin_guardrails() -> None:
    for name, factory in (
        ("pii", _guardrail_pii),
        ("blocklist", _guardrail_blocklist),
        ("llm_judge", _guardrail_llm_judge),
    ):
        if not GUARDRAILS.has(name):
            GUARDRAILS.register(name, factory, category="safety")


def compile_guardrails(specs: list[GuardrailSpec]) -> list[Guardrail]:
    """All guardrails as ``Guardrail`` objects (e.g. for output-side wiring)."""
    return [GUARDRAILS.get(g.type)(g) for g in specs]


# =============================================================================
# input pre-flight engine
# =============================================================================
@dataclass
class CompiledInputRule:
    spec: GuardrailSpec
    guardrail: Guardrail


def compile_input_rules(specs: list[GuardrailSpec]) -> list[CompiledInputRule]:
    """Compile only the input-side guardrails for the pre-flight engine."""
    return [
        CompiledInputRule(spec=g, guardrail=GUARDRAILS.get(g.type)(g))
        for g in specs
        if g.on_input
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
    """Run input guardrails before any model call.

    Threads ``text`` through each rule: a redact rule rewrites it, a block rule
    short-circuits with a refusal.
    """
    current = text
    triggered: list[GuardrailHit] = []
    for rule in rules:
        try:
            result = rule.guardrail.check_input(current)
        except Exception:
            continue  # judge/model failure → fail open
        if not result.triggered:
            continue
        if result.blocked:
            triggered.append(GuardrailHit(rule.spec.type, rule.spec.action))
            return InputOutcome(False, current, result.refusal or _DEFAULT_REFUSAL, triggered)
        if result.new_text is not None and result.new_text != current:
            triggered.append(GuardrailHit(rule.spec.type, rule.spec.action))
            current = result.new_text
        else:
            triggered.append(GuardrailHit(rule.spec.type, rule.spec.action))
    return InputOutcome(True, current, None, triggered)


__all__ = [
    "Guardrail",
    "BlocklistGuardrail",
    "PIIGuardrail",
    "LLMJudgeGuardrail",
    "InputCheck",
    "CompiledInputRule",
    "GuardrailHit",
    "InputOutcome",
    "compile_guardrails",
    "compile_input_rules",
    "run_input_guardrails",
    "register_builtin_guardrails",
]
