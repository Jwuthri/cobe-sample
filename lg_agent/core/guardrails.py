"""Guardrails — config rules compiled to middleware + a pre-flight input engine.

Two consumers share one set of rule objects (so they can't drift):

  * :func:`compile_guardrail_middleware` — attach rules as middleware to an agent
    via ``build_agent`` (output-side checks run here).
  * :func:`compile_input_rules` + :func:`run_input_guardrails` — run the SAME
    middleware instances' ``before_model`` as a pre-flight step BEFORE the
    orchestrator's first model call. A refusal is instant; a redaction rewrites the
    user text before it enters the transcript, so every downstream model sees only
    clean input.

Three rule types are registered: ``pii`` (built-in ``PIIMiddleware``), ``blocklist``
(phrase/regex), and ``llm_judge`` (a small model judges a natural-language policy).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    PIIDetectionError,
    PIIMiddleware,
)
from langchain.agents.middleware.types import hook_config
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from lg_agent.core.config import GuardrailSpec
from lg_agent.core.registry import GUARDRAILS

_DEFAULT_REFUSAL = "I'm not able to help with that request."


# =============================================================================
# helpers
# =============================================================================
def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)


def _last_text(messages: list[Any], cls: type) -> str:
    for m in reversed(messages or []):
        if isinstance(m, cls):
            return _message_text(m)
    return ""


# =============================================================================
# blocklist
# =============================================================================
class BlocklistGuardrail(AgentMiddleware):
    """Phrase / regex guardrail. Blocks on input (short-circuit) or output (replace)."""

    def __init__(
        self,
        phrases: list[str] | None = None,
        patterns: list[str] | None = None,
        action: str = "block",
        message: str | None = None,
        on_input: bool = True,
        on_output: bool = False,
    ) -> None:
        super().__init__()
        self.phrases = tuple(p.lower() for p in (phrases or []))
        self.patterns = tuple(re.compile(p, re.IGNORECASE) for p in (patterns or []))
        self.action = action
        self.message = message or _DEFAULT_REFUSAL
        self.on_input = on_input
        self.on_output = on_output

    def _matches(self, text: str) -> bool:
        lowered = text.lower()
        if any(phrase in lowered for phrase in self.phrases):
            return True
        return any(pattern.search(text) for pattern in self.patterns)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        if self.on_input and self._matches(_last_text(state.get("messages", []), HumanMessage)):
            return {"messages": [AIMessage(content=self.message)], "jump_to": "end"}
        return None

    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        if not self.on_output:
            return None
        messages = state.get("messages", [])
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai is None or not self._matches(_message_text(last_ai)):
            return None
        # Replace the violating message BY ID so the offending text never persists.
        return {"messages": [AIMessage(content=self.message, id=last_ai.id)]}


# =============================================================================
# llm_judge
# =============================================================================
class _PolicyVerdict(BaseModel):
    violates: bool = Field(description="True if the message breaks the policy.")
    reason: str = Field(default="", description="Short reason for the decision.")


_POLICY_JUDGE_SYSTEM = """You are a content-policy checker.
Decide if the message violates this policy:

<policy>
{policy}
</policy>

Judge by meaning, not keywords. Set violates=true only if it clearly breaks the policy."""


class LLMGuardrail(AgentMiddleware):
    """Semantic guardrail: a small model judges the text against a policy.

    On a judge error it FAILS OPEN (does not block) — demo availability over
    strictness; tune for your risk tolerance. ``judge_factory`` is injectable for
    tests (so no real model call is made).
    """

    def __init__(
        self,
        policy: str,
        model: str = "openai:gpt-4.1-mini",
        action: str = "block",
        message: str | None = None,
        on_input: bool = True,
        on_output: bool = False,
        judge_factory: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.model = model
        self.action = action
        self.message = message or _DEFAULT_REFUSAL
        self.on_input = on_input
        self.on_output = on_output
        self._judge_factory = judge_factory
        self._judge_cache: Any = None

    def _judge(self) -> Any:
        if self._judge_cache is None:
            if self._judge_factory is not None:
                self._judge_cache = self._judge_factory()
            else:
                # No forced temperature (gpt-5-family rejects non-default values);
                # tag "nostream" so a judge call can never enter a token stream.
                self._judge_cache = (
                    init_chat_model(self.model)
                    .with_config(tags=["nostream"])
                    .with_structured_output(_PolicyVerdict)
                )
        return self._judge_cache

    def _violates(self, text: str) -> bool:
        if not text.strip():
            return False
        try:
            verdict: _PolicyVerdict = self._judge().invoke(
                [
                    SystemMessage(content=_POLICY_JUDGE_SYSTEM.format(policy=self.policy)),
                    HumanMessage(content=text),
                ]
            )
            return bool(getattr(verdict, "violates", False))
        except Exception:
            return False  # fail open

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        if self.on_input and self._violates(_last_text(state.get("messages", []), HumanMessage)):
            return {"messages": [AIMessage(content=self.message)], "jump_to": "end"}
        return None

    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        if not self.on_output:
            return None
        messages = state.get("messages", [])
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai is None or not self._violates(_message_text(last_ai)):
            return None
        return {"messages": [AIMessage(content=self.message, id=last_ai.id)]}


# =============================================================================
# rule factories (type -> factory(GuardrailSpec) -> AgentMiddleware)
# =============================================================================
def _guardrail_pii(gr: GuardrailSpec) -> AgentMiddleware:
    entity = gr.params.get("entity", "email")
    strategy = gr.action if gr.action in ("block", "redact", "mask", "hash") else "redact"
    if strategy == "flag":  # not a valid PII strategy
        strategy = "redact"
    return PIIMiddleware(
        entity,
        strategy=strategy,
        apply_to_input=gr.on_input,
        apply_to_output=gr.on_output,
        apply_to_tool_results=bool(gr.params.get("tool_results", False)),
    )


def _guardrail_blocklist(gr: GuardrailSpec) -> AgentMiddleware:
    return BlocklistGuardrail(
        phrases=gr.params.get("phrases"),
        patterns=gr.params.get("patterns"),
        action=gr.action,
        message=gr.message,
        on_input=gr.on_input,
        on_output=gr.on_output,
    )


def _guardrail_llm_judge(gr: GuardrailSpec) -> AgentMiddleware:
    policy = gr.params.get("policy")
    if not policy:
        raise ValueError("llm_judge guardrail requires params.policy")
    return LLMGuardrail(
        policy=policy,
        model=gr.params.get("model", "openai:gpt-4.1-mini"),
        action=gr.action,
        message=gr.message,
        on_input=gr.on_input,
        on_output=gr.on_output,
        judge_factory=gr.params.get("judge_factory"),
    )


def register_builtin_guardrails() -> None:
    for name, factory in (
        ("pii", _guardrail_pii),
        ("blocklist", _guardrail_blocklist),
        ("llm_judge", _guardrail_llm_judge),
    ):
        if not GUARDRAILS.has(name):
            GUARDRAILS.register(name, factory, category="safety")


# =============================================================================
# build-time + pre-flight surfaces
# =============================================================================
def compile_guardrail_middleware(specs: list[GuardrailSpec]) -> list[AgentMiddleware]:
    """All guardrails as middleware (used by build_agent for sub-agents/writer)."""
    return [GUARDRAILS.get(g.type)(g) for g in specs]


@dataclass
class CompiledInputRule:
    spec: GuardrailSpec
    middleware: AgentMiddleware


def compile_input_rules(specs: list[GuardrailSpec]) -> list[CompiledInputRule]:
    """Compile only the input-side guardrails for the pre-flight engine."""
    return [
        CompiledInputRule(spec=g, middleware=GUARDRAILS.get(g.type)(g))
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

    Threads ``text`` through each rule's ``before_model``: a redact rule rewrites
    it, a block rule short-circuits with a refusal. Reuses the same middleware
    instances as the build-time path, so the two can't diverge.
    """
    current = text
    triggered: list[GuardrailHit] = []
    for rule in rules:
        mw = rule.middleware
        spec = rule.spec
        try:
            result = mw.before_model({"messages": [HumanMessage(content=current)]}, None)
        except PIIDetectionError:
            triggered.append(GuardrailHit(spec.type, spec.action))
            return InputOutcome(False, current, spec.message or _DEFAULT_REFUSAL, triggered)
        except Exception:
            continue  # judge/model failure → fail open
        if not result:
            continue
        if result.get("jump_to") == "end":
            refusal = spec.message or _last_text(result.get("messages", []), AIMessage) or _DEFAULT_REFUSAL
            triggered.append(GuardrailHit(spec.type, spec.action))
            return InputOutcome(False, current, refusal, triggered)
        new_messages = result.get("messages")
        if new_messages:
            redacted = _last_text(new_messages, HumanMessage)
            if redacted and redacted != current:
                triggered.append(GuardrailHit(spec.type, spec.action))
                current = redacted
    return InputOutcome(True, current, None, triggered)


__all__ = [
    "BlocklistGuardrail",
    "LLMGuardrail",
    "CompiledInputRule",
    "GuardrailHit",
    "InputOutcome",
    "compile_guardrail_middleware",
    "compile_input_rules",
    "run_input_guardrails",
    "register_builtin_guardrails",
]
