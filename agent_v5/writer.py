"""The optional writer — used ONLY by the ``router`` variant.

When the supervisor is a pure router (it emits ``DONE`` instead of prose), this
module composes the user-facing message — the same role v4's ``writer`` node
played. To stay faithful we reuse v4's writer verbatim: its ``WRITER_SYSTEM``
prompt, its mode selection, and its payload builder. v4's helpers expect an
``AgentState``, so we feed them a tiny duck-typed shim exposing just the four
attributes they read (``step_results``, ``cart``, ``messages``,
``last_user_message()``).

The ``speaking`` variant does NOT import this — that's the whole point of the
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_v4.llm import writer_model_name
from agent_v4.step_result import StepResult
from agent_v4.writer import WRITER_SYSTEM, _build_writer_payload
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


@dataclass
class _WriterState:
    """Minimal duck-type of ``AgentState`` for v4's writer payload helpers."""

    step_results: list[StepResult]
    cart: object
    messages: list[BaseMessage]

    def last_user_message(self) -> str:
        for m in reversed(self.messages):
            if isinstance(m, HumanMessage):
                return str(m.content)
        return ""


def compose_reply(
    messages: list[BaseMessage], step_results: list[StepResult], cart
) -> tuple[str, dict[str, int]]:
    """Run v4's writer over the accumulated results.

    Returns ``(prose, usage)`` where usage is ``{input_tokens, output_tokens,
    llm_calls}`` for this single writer call — so the orchestrator can fold the
    writer's cost into the per-turn total for the comparison.
    """
    shim = _WriterState(step_results=step_results, cart=cart, messages=messages)
    payload, _mode = _build_writer_payload(shim)
    chat = ChatOpenAI(model=writer_model_name(), temperature=0.3)
    resp = chat.invoke(
        [SystemMessage(content=WRITER_SYSTEM), HumanMessage(content=payload)]
    )
    text = (resp.content or "").strip() if isinstance(resp.content, str) else str(resp.content)
    um = getattr(resp, "usage_metadata", None) or {}
    usage = {
        "input_tokens": int(um.get("input_tokens", 0) or 0),
        "output_tokens": int(um.get("output_tokens", 0) or 0),
        "cached_tokens": int((um.get("input_token_details") or {}).get("cache_read", 0) or 0),
        "llm_calls": 1,
    }
    return (text or "(no response)"), usage


__all__ = ["compose_reply"]
