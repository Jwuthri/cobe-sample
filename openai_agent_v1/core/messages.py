"""A tiny framework-agnostic message vocabulary.

agent_v4_1 threaded LangChain ``HumanMessage`` / ``AIMessage`` / ``ToolMessage``
objects through the transcript, extractors, writer payload, and trace renderer.
This clean-room port runs on the OpenAI Agents SDK (whose run items are a
different shape), so it defines ONE small ``Msg`` record those same components
share. ``items_to_msgs`` converts a sub-agent ``RunResult.new_items`` list into
this vocabulary, pairing each tool call with its output so the (verbatim)
extractors can read ``role == "tool"`` + ``name`` + ``content`` exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

Role = Literal["human", "ai", "system", "tool"]


@dataclass
class Msg:
    """One message. ``role`` mirrors the v4_1 chips (human/ai/system/tool)."""

    role: Role
    content: str = ""
    name: str | None = None  # tool name, for role == "tool"
    tool_calls: list[dict] = field(default_factory=list)  # for role == "ai"
    blocks: list[dict] = field(default_factory=list)  # rich-reply blocks, for role == "ai"


def human(content: str) -> Msg:
    return Msg(role="human", content=content)


def ai(content: str, tool_calls: list[dict] | None = None, blocks: list[dict] | None = None) -> Msg:
    return Msg(role="ai", content=content, tool_calls=list(tool_calls or []), blocks=list(blocks or []))


def system(content: str) -> Msg:
    return Msg(role="system", content=content)


def tool_msg(name: str, content: str) -> Msg:
    return Msg(role="tool", name=name, content=content)


# =============================================================================
# SDK input conversion (Msg list -> Responses API input items)
# =============================================================================
_ROLE_TO_INPUT = {"human": "user", "ai": "assistant", "system": "system"}


def msgs_to_input(messages: Iterable[Msg]) -> list[dict[str, Any]]:
    """Render a ``Msg`` transcript as OpenAI Agents SDK input items.

    Tool messages are dropped (they belong to a finished sub-agent run, not the
    orchestrator's transcript). Empty assistant turns are dropped too.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = _ROLE_TO_INPUT.get(m.role)
        if role is None:
            continue
        if role == "assistant" and not (m.content or "").strip():
            continue
        out.append({"role": role, "content": m.content})
    return out


# =============================================================================
# SDK output conversion (RunResult.new_items -> Msg list)
# =============================================================================
def _output_text(raw_item: Any) -> str:
    """Best-effort text from a tool-output raw_item (dict or object)."""
    if isinstance(raw_item, dict):
        out = raw_item.get("output")
    else:
        out = getattr(raw_item, "output", None)
    if isinstance(out, dict):  # function_call_output sometimes wraps {type, text}
        return str(out.get("text", out.get("output", out)))
    return "" if out is None else str(out)


def items_to_msgs(new_items: Iterable[Any]) -> list[Msg]:
    """Convert an SDK ``RunResult.new_items`` sequence into ``Msg`` records.

    Pairs each ``ToolCallItem`` (which carries the tool name) with its matching
    ``ToolCallOutputItem`` (which carries the output) by ``call_id``, so a tool
    message ends up with both ``name`` and ``content`` — what the extractors read.
    """
    name_by_call: dict[str, str] = {}
    msgs: list[Msg] = []
    for item in new_items or []:
        itype = getattr(item, "type", None)
        raw = getattr(item, "raw_item", None)
        if itype == "tool_call_item":
            call_id = getattr(raw, "call_id", None) or (
                raw.get("call_id") if isinstance(raw, dict) else None
            )
            name = getattr(raw, "name", None) or (
                raw.get("name") if isinstance(raw, dict) else None
            )
            args = getattr(raw, "arguments", None)
            if call_id and name:
                name_by_call[call_id] = name
            msgs.append(Msg(role="ai", content="", tool_calls=[{"name": name, "args": args}]))
        elif itype == "tool_call_output_item":
            call_id = (
                raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", None)
            )
            name = name_by_call.get(call_id or "", "")
            content = getattr(item, "output", None)
            if content is None:
                content = _output_text(raw)
            msgs.append(Msg(role="tool", name=name, content=str(content)))
        elif itype == "message_output_item":
            from agents.items import ItemHelpers  # local import; SDK-only path

            try:
                text = ItemHelpers.text_message_output(item)
            except Exception:  # pragma: no cover
                text = ""
            msgs.append(Msg(role="ai", content=text))
    return msgs


__all__ = [
    "Msg",
    "Role",
    "human",
    "ai",
    "system",
    "tool_msg",
    "msgs_to_input",
    "items_to_msgs",
]
