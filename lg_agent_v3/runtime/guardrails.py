"""Guardrails — real per-agent safety rules as ``before_agent``/``after_agent`` middleware.

Every agent (orchestrator, each sub-agent, the writer) can declare its own guardrails;
``compile_guardrails(specs, agent_name)`` turns them into middleware attached to *that*
agent's graph. The hooks fire **once per agent run** (one input check on entry via
``before_agent``, one output check on exit via ``after_agent``) — not per model step.

Three rule types:
  * ``blocklist`` — phrase/regex match → block (input: short-circuit to a refusal;
    output: replace the offending message by id).
  * ``llm_judge`` — a small model judges a natural-language ``policy`` → block. Fails
    OPEN on judge error (availability over strictness). ``judge_factory`` is injectable
    for offline tests.
  * ``pii`` — regex redaction (input rewrites the user text; output scrubs the reply).
    The same redactor backs the session-level input sanitizer (:func:`redact_input`),
    so the two can't drift.

**Surfacing.** When a guardrail fires it records a :class:`GuardrailHit` on the shared
context (``deps.guardrail_hits``) and emits a ``{type:"guardrail"}`` UI event. The
middleware *signals*; delivery happens elsewhere (the session routes an orchestrator
block to the writer; ``run_subagent`` turns a sub-agent block into a flagged step).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import hook_config
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from lg_agent_v3.runtime import events

_DEFAULT_REFUSAL = "I'm not able to help with that request."


# --------------------------------------------------------------------------- #
# spec + hit
# --------------------------------------------------------------------------- #
@dataclass
class GuardrailSpec:
    """A declarative guardrail rule (the JSON-friendly config shape)."""

    type: str  # "blocklist" | "pii" | "llm_judge"
    action: str = "block"  # "block" | "redact"
    on_input: bool = True
    on_output: bool = False
    message: str | None = None
    params: dict = field(default_factory=dict)


@dataclass
class GuardrailHit:
    agent: str
    type: str
    action: str
    side: str  # "input" | "output"
    message: str | None = None


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(b.get("text", "")) if isinstance(b, dict) else str(b)
            for b in content
            if (isinstance(b, dict) and b.get("type") == "text") or isinstance(b, str)
        ]
        return " ".join(parts)
    return str(content)


def _last(messages: list[Any], cls: type) -> Any | None:
    for m in reversed(messages or []):
        if isinstance(m, cls):
            return m
    return None


def _last_text(messages: list[Any], cls: type) -> str:
    m = _last(messages, cls)
    return _message_text(m) if m is not None else ""


# --------------------------------------------------------------------------- #
# base — records a hit + emits a UI event when a rule fires
# --------------------------------------------------------------------------- #
class _Guard(AgentMiddleware):
    def __init__(self, spec: GuardrailSpec, agent_name: str = "") -> None:
        super().__init__()
        self.spec = spec
        self.agent_name = agent_name or "agent"

    def _hit(self, runtime: Any, side: str) -> None:
        ctx = getattr(runtime, "context", None)
        if ctx is None:
            return
        hit = GuardrailHit(self.agent_name, self.spec.type, self.spec.action, side, self.spec.message)
        hits = getattr(ctx, "guardrail_hits", None)
        if hits is not None:
            hits.append(hit)
        if hasattr(ctx, "emit"):
            ctx.emit(events.guardrail(f"{self.agent_name}:{side}", self.spec.type, self.spec.action))


# --------------------------------------------------------------------------- #
# blocklist
# --------------------------------------------------------------------------- #
class BlocklistGuardrail(_Guard):
    def __init__(self, spec: GuardrailSpec, agent_name: str = "") -> None:
        super().__init__(spec, agent_name)
        self.phrases = tuple(p.lower() for p in (spec.params.get("phrases") or []))
        self.patterns = tuple(re.compile(p, re.IGNORECASE) for p in (spec.params.get("patterns") or []))
        self.message = spec.message or _DEFAULT_REFUSAL

    def _matches(self, text: str) -> bool:
        low = text.lower()
        return any(p in low for p in self.phrases) or any(p.search(text) for p in self.patterns)

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        if self.spec.on_input and self._matches(_last_text(state.get("messages", []), HumanMessage)):
            self._hit(runtime, "input")
            return {"messages": [AIMessage(content=self.message)], "jump_to": "end"}
        return None

    def after_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        if not self.spec.on_output:
            return None
        last = _last(state.get("messages", []), AIMessage)
        if last is not None and self._matches(_message_text(last)):
            self._hit(runtime, "output")
            return {"messages": [AIMessage(content=self.message, id=last.id)]}
        return None


# --------------------------------------------------------------------------- #
# llm_judge
# --------------------------------------------------------------------------- #
class _PolicyVerdict(BaseModel):
    violates: bool = Field(description="True if the message breaks the policy.")
    reason: str = Field(default="", description="Short reason for the decision.")


_POLICY_JUDGE_SYSTEM = """You are a content-policy checker.
Decide if the message violates this policy:

<policy>
{policy}
</policy>

Judge by meaning, not keywords. Set violates=true only if it clearly breaks the policy."""


class LLMGuardrail(_Guard):
    def __init__(self, spec: GuardrailSpec, agent_name: str = "") -> None:
        super().__init__(spec, agent_name)
        self.policy = spec.params.get("policy")
        if not self.policy:
            raise ValueError("llm_judge guardrail requires params.policy")
        self.model = spec.params.get("model", "openai:gpt-4.1-mini")
        self.message = spec.message or _DEFAULT_REFUSAL
        self._judge_factory: Callable[[], Any] | None = spec.params.get("judge_factory")
        self._judge_cache: Any = None

    def _judge(self) -> Any:
        if self._judge_cache is None:
            self._judge_cache = (
                self._judge_factory()
                if self._judge_factory is not None
                else init_chat_model(self.model).with_config(tags=["nostream"]).with_structured_output(_PolicyVerdict)
            )
        return self._judge_cache

    def _violates(self, text: str) -> bool:
        if not text.strip():
            return False
        try:
            verdict = self._judge().invoke(
                [SystemMessage(content=_POLICY_JUDGE_SYSTEM.format(policy=self.policy)), HumanMessage(content=text)]
            )
            return bool(getattr(verdict, "violates", False))
        except Exception:  # noqa: BLE001
            return False  # fail open

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        if self.spec.on_input and self._violates(_last_text(state.get("messages", []), HumanMessage)):
            self._hit(runtime, "input")
            return {"messages": [AIMessage(content=self.message)], "jump_to": "end"}
        return None

    def after_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        if not self.spec.on_output:
            return None
        last = _last(state.get("messages", []), AIMessage)
        if last is not None and self._violates(_message_text(last)):
            self._hit(runtime, "output")
            return {"messages": [AIMessage(content=self.message, id=last.id)]}
        return None


# --------------------------------------------------------------------------- #
# pii — regex redaction (also backs the session-level input sanitizer)
# --------------------------------------------------------------------------- #
class PiiRedactGuardrail(_Guard):
    def __init__(self, spec: GuardrailSpec, agent_name: str = "") -> None:
        super().__init__(spec, agent_name)
        pats = spec.params.get("patterns") or [r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"]  # email by default
        self.patterns = tuple(re.compile(p) for p in pats)
        self.placeholder = spec.params.get("placeholder", "[redacted]")

    def redact(self, text: str) -> str:
        out = text
        for p in self.patterns:
            out = p.sub(self.placeholder, out)
        return out

    def before_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        if not self.spec.on_input:
            return None
        msgs = list(state.get("messages", []))
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], HumanMessage):
                red = self.redact(_message_text(msgs[i]))
                if red != _message_text(msgs[i]):
                    self._hit(runtime, "input")
                    msgs[i] = HumanMessage(content=red, id=msgs[i].id)  # replace by id
                    return {"messages": msgs}
                break
        return None

    def after_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        if not self.spec.on_output:
            return None
        last = _last(state.get("messages", []), AIMessage)
        if last is not None:
            red = self.redact(_message_text(last))
            if red != _message_text(last):
                self._hit(runtime, "output")
                return {"messages": [AIMessage(content=red, id=last.id)]}
        return None


# --------------------------------------------------------------------------- #
# compile + session-level input redactor
# --------------------------------------------------------------------------- #
_FACTORIES: dict[str, type[_Guard]] = {
    "blocklist": BlocklistGuardrail,
    "llm_judge": LLMGuardrail,
    "pii": PiiRedactGuardrail,
}


def compile_guardrails(specs: list[GuardrailSpec] | None, agent_name: str = "") -> list[AgentMiddleware]:
    """Compile guardrail specs into per-agent middleware (attached at build time)."""
    return [_FACTORIES[s.type](s, agent_name) for s in (specs or [])]


def redact_input(specs: list[GuardrailSpec] | None, text: str) -> tuple[str, list[GuardrailHit]]:
    """Session-level input sanitizer: apply ``pii`` redact rules BEFORE the text enters
    the transcript (so the raw value never hits storage or downstream agents). Blocking
    rules are NOT run here — they live on the agents (`before_agent`)."""
    hits: list[GuardrailHit] = []
    for s in specs or []:
        if s.type == "pii" and s.on_input:
            red = PiiRedactGuardrail(s).redact(text)
            if red != text:
                hits.append(GuardrailHit("session", s.type, s.action, "input", s.message))
                text = red
    return text, hits


__all__ = [
    "GuardrailSpec",
    "GuardrailHit",
    "BlocklistGuardrail",
    "LLMGuardrail",
    "PiiRedactGuardrail",
    "compile_guardrails",
    "redact_input",
]
