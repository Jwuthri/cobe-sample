"""Outer-graph state (Pydantic, raw StateGraph — Command-routed).

``active_sop`` / ``target_sop`` are leaf ids (plain strings, see
:mod:`agent_v4.ids`). v2 used a hard-coded ``SOPName`` enum here; v4 keeps
the leaf vocabulary data-driven so adding a leaf never edits this file.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from agent_v4.checkout import Cart
from agent_v4.step_result import StepResult
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class HandoverReason(str, Enum):
    SWITCH_SOP = "switch_sop"
    CLARIFY = "clarify"
    ESCALATE_HUMAN = "escalate_human"
    VALIDATION_FAILURE = "validation_failure"
    GATE_FAILED = "gate_failed"


class HandoverSignal(BaseModel):
    reason: HandoverReason
    from_node: str
    target_sop: str | None = None
    note: str = ""


class ValidationError(BaseModel):
    code: Literal["empty", "too_long", "placeholder_leak", "unsafe", "gate"]
    detail: str


MAX_VALIDATOR_RETRIES = 2


def _append_strs(left: list[str] | None, right: list[str] | None) -> list[str]:
    out = list(left or [])
    for x in right or []:
        if x not in out:
            out.append(x)
    return out


def _append_step_results(
    left: list[StepResult] | None, right: list[StepResult] | None
) -> list[StepResult]:
    return list(left or []) + list(right or [])


class AgentState(BaseModel):
    # conversation — we accumulate LangChain BaseMessage objects so the
    # inner subagents can be invoked with the full history.
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    user_id: str = "anonymous"
    session_id: str = "session"

    # routing — a leaf id (e.g. "checkout") or None
    active_sop: str | None = None
    last_handover: HandoverSignal | None = None

    # checkout state (the rich cart)
    cart: Cart = Field(default_factory=Cart)

    # skills loaded INSIDE the checkout subagent. We propagate across
    # turns of the same session so the model doesn't re-load skills.
    skills_loaded: Annotated[list[str], _append_strs] = Field(default_factory=list)

    # v4 loop state
    step_results: Annotated[list[StepResult], _append_step_results] = Field(default_factory=list)
    iteration: int = 0

    # responder / validator
    draft_response: str | None = None
    validation_errors: list[ValidationError] = Field(default_factory=list)
    response_attempts: int = 0

    done: bool = False

    # ---- helpers ----
    def last_user_message(self) -> str:
        for m in reversed(self.messages):
            if isinstance(m, BaseMessage) and m.type == "human":
                return str(m.content)
            # dict fallback (some construction paths)
            if isinstance(m, dict) and m.get("type") == "human":
                return str(m.get("content", ""))
        return ""
