"""Input guardrails — a pre-flight gate that runs BEFORE any model call.

Keeping safety checks off the token path is what lets the writer stream freely: a
refusal here is instant (no model call), and a redaction rewrites the user text
before it ever enters the transcript. The engine is deliberately small — a list
of compiled rules, each a ``(name, action, check)`` — and defaults to empty, so a
plain session has zero overhead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

Action = Literal["block", "redact"]


@dataclass(frozen=True)
class CompiledInputRule:
    type: str
    action: Action
    # check(text) -> (passed, transformed_text). For "block", passed=False trips it;
    # for "redact", transformed_text is the rewritten input.
    check: Callable[[str], tuple[bool, str]]
    message: str = "I can't help with that request."


@dataclass
class GuardrailHit:
    type: str
    action: Action


@dataclass
class GuardrailOutcome:
    allowed: bool = True
    text: str = ""
    refusal: str | None = None
    triggered: list[GuardrailHit] = field(default_factory=list)


def run_input_guardrails(rules: list[CompiledInputRule], text: str) -> GuardrailOutcome:
    """Apply each rule in order. A tripped ``block`` short-circuits to a refusal."""
    outcome = GuardrailOutcome(allowed=True, text=text)
    for rule in rules:
        passed, transformed = rule.check(outcome.text)
        if rule.action == "block" and not passed:
            outcome.allowed = False
            outcome.refusal = rule.message
            outcome.triggered.append(GuardrailHit(rule.type, rule.action))
            return outcome
        if rule.action == "redact" and transformed != outcome.text:
            outcome.text = transformed
            outcome.triggered.append(GuardrailHit(rule.type, rule.action))
    return outcome


def blocklist_rule(words: list[str], message: str = "I can't help with that request.") -> CompiledInputRule:
    """Refuse if the input contains any blocked word (case-insensitive)."""
    pattern = re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)

    def check(text: str) -> tuple[bool, str]:
        return (pattern.search(text) is None, text)

    return CompiledInputRule(type="blocklist", action="block", check=check, message=message)


__all__ = [
    "CompiledInputRule",
    "GuardrailHit",
    "GuardrailOutcome",
    "run_input_guardrails",
    "blocklist_rule",
]
