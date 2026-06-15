"""Input guardrails — a pre-flight gate that runs BEFORE any model call.

A turn threads the user's text through each rule in order. A ``block`` rule
short-circuits the whole turn with a refusal; a ``redact`` rule rewrites the text
before it enters the transcript, so every downstream agent sees only clean input.

Two deterministic rule types ship here (no model needed): a phrase/regex
``Blocklist`` and a regex ``PiiRedact``. The shopping session defaults to an empty
rule list — guardrails are opt-in, demonstrated rather than imposed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

_DEFAULT_REFUSAL = "I'm not able to help with that request."


class InputRule(Protocol):
    type: str

    def apply(self, text: str) -> tuple[str | None, str]:
        """Return ``(refusal, text)``. A non-None refusal blocks; ``text`` may be rewritten."""
        ...


class Blocklist:
    type = "blocklist"

    def __init__(
        self,
        phrases: list[str] | None = None,
        patterns: list[str] | None = None,
        message: str | None = None,
    ) -> None:
        self.phrases = tuple(p.lower() for p in (phrases or []))
        self.patterns = tuple(re.compile(p, re.IGNORECASE) for p in (patterns or []))
        self.message = message or _DEFAULT_REFUSAL

    def apply(self, text: str) -> tuple[str | None, str]:
        low = text.lower()
        if any(p in low for p in self.phrases) or any(p.search(text) for p in self.patterns):
            return self.message, text
        return None, text


class PiiRedact:
    type = "pii"

    def __init__(self, patterns: list[str] | None = None, placeholder: str = "[redacted]") -> None:
        default = [r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"]  # email
        self.patterns = tuple(re.compile(p) for p in (patterns or default))
        self.placeholder = placeholder

    def apply(self, text: str) -> tuple[str | None, str]:
        new = text
        for p in self.patterns:
            new = p.sub(self.placeholder, new)
        return None, new


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


def run_input_guardrails(rules: list[InputRule], text: str) -> InputOutcome:
    current = text
    triggered: list[GuardrailHit] = []
    for rule in rules:
        refusal, rewritten = rule.apply(current)
        if refusal is not None:
            triggered.append(GuardrailHit(rule.type, "block"))
            return InputOutcome(False, current, refusal, triggered)
        if rewritten != current:
            triggered.append(GuardrailHit(rule.type, "redact"))
            current = rewritten
    return InputOutcome(True, current, None, triggered)
