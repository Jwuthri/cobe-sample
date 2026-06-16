"""Agent-as-tool — the one wrapper that turns a worker agent into a delegate.

In this topology the orchestrator's "tools" ARE the worker agents. The SDK gives
us ``Agent.as_tool(tool_name, tool_description, custom_output_extractor=...,
is_enabled=...)`` natively, so the whole helper here is a single
:func:`build_worker_tool` that:

  1. wires ``custom_output_extractor`` to a worker-specific function which
     receives the worker's :class:`agents.RunResult`, parses its ``new_items``
     into a :class:`StepResult`, appends it to the shared context's ``steps``
     list, AND returns the terse one-line summary the orchestrator reads back;
  2. wires ``is_enabled`` for tools that should be hidden in some states (the
     empty-cart guard on checkout).

**Context isolation.** A worker never sees the conversation. The orchestrator is
the sole reader of the transcript: it resolves references ("the green one", "add
it") into a concrete, self-contained ``query`` and ``as_tool`` passes only that to
the worker run. A worker operates on ``(query + the shared cart)`` — the cart is
its memory, not the chat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agents import Agent, RunContextWrapper
from agents.items import ToolCallItem, ToolCallOutputItem
from agents.result import RunResult

from agent_openai_sdk_v1.runtime.context import ShoppingContext
from agent_openai_sdk_v1.runtime.events import step as _step_event, tool_end as _tool_end_event, tool_start as _tool_start_event
from agent_openai_sdk_v1.runtime.step import StepResult

# A worker turns its run's items into a StepResult.
Extractor = Callable[[ShoppingContext, list[Any]], StepResult]
Summarizer = Callable[[StepResult, ShoppingContext], str]


@dataclass(frozen=True)
class Worker:
    """A worker agent + the small hooks that make it concrete.

    The ``agent`` is a plain SDK ``Agent``; the hooks translate between the shared
    domain state and the orchestrator. Only ``extract`` is required.
    """

    name: str  # orchestrator-facing tool name AND the StepResult.sop
    agent: Agent  # the OpenAI Agents SDK sub-agent
    description: str  # the delegate-tool description (the orchestrator's routing surface)
    extract: Extractor  # run items → StepResult
    prompt: str = ""  # static instructions (shown in the debug trace)
    block: str | None = None  # writer block kind this worker produces
    summarize: Summarizer | None = None  # default: sr.summary
    is_enabled: Callable[[RunContextWrapper[ShoppingContext], Any], bool] | None = None


def build_worker_tool(worker: Worker):
    """Turn a :class:`Worker` into a delegate tool the orchestrator can call.

    The returned object is the SDK's ``FunctionTool`` (from ``Agent.as_tool``) —
    passing the shared ``ShoppingContext`` through and folding the worker's run
    output into a :class:`StepResult` on the way out.
    """

    async def _extractor(result: RunResult) -> str:
        # The SDK passes ``RunResult`` here; ``result.context_wrapper.context`` is
        # the parent's ShoppingContext (the SAME one threaded through the
        # orchestrator → workers → tools — one live cart end-to-end).
        ctx: ShoppingContext = result.context_wrapper.context
        items = list(result.new_items)
        # Stash inner tool_start / tool_end events for the session to drain inline
        # with the orchestrator's outer stream (this is how the live UI sees what
        # the worker did, e.g. add_item / set_address).
        for ev in _inner_tool_events(items):
            ctx.pending_events.append(ev)
        sr = worker.extract(ctx, items)
        ctx.steps.append(sr)
        ctx.pending_events.append(_step_event(sr))
        return worker.summarize(sr, ctx) if worker.summarize else sr.summary

    return worker.agent.as_tool(
        tool_name=worker.name,
        tool_description=worker.description,
        custom_output_extractor=_extractor,
        is_enabled=worker.is_enabled if worker.is_enabled is not None else True,
    )


# --------------------------------------------------------------------------- #
# RunItem helpers (consumed by worker extractors to read what their tools returned)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolReturn:
    name: str
    content: str


def tool_returns(items: list[Any]) -> list[ToolReturn]:
    """Extract ``(tool_name, content)`` pairs from a worker's ``new_items``.

    Workers use this in their ``extract`` hook to read what their tools returned
    (e.g. the catalog lines ``search_products`` produced) without re-deriving state.
    A ``ToolCallOutputItem`` carries the result but no tool name — we look the
    name up via the matching ``ToolCallItem`` (same ``call_id``).
    """
    name_by_call_id: dict[str, str] = {}
    for it in items or []:
        if isinstance(it, ToolCallItem) and it.call_id and it.tool_name:
            name_by_call_id[it.call_id] = it.tool_name

    out: list[ToolReturn] = []
    for it in items or []:
        if not isinstance(it, ToolCallOutputItem):
            continue
        name = name_by_call_id.get(it.call_id or "", "tool")
        out.append(ToolReturn(name, str(it.output) if it.output is not None else ""))
    return out


def _inner_tool_events(items: list[Any]) -> list[dict]:
    """Walk a worker's ``new_items`` → ordered ``tool_start`` / ``tool_end`` events.

    The tool name lives on the ``ToolCallItem``; the call output lives on the
    matching ``ToolCallOutputItem`` (paired by ``call_id``). We emit start/end
    pairs in document order so the live UI renders them as the worker did them.
    """
    import json

    name_by_call_id: dict[str, str] = {}
    args_by_call_id: dict[str, dict] = {}
    for it in items or []:
        if isinstance(it, ToolCallItem) and it.call_id and it.tool_name:
            name_by_call_id[it.call_id] = it.tool_name
            raw_args = getattr(it.raw_item, "arguments", "") or ""
            try:
                parsed = json.loads(raw_args) if raw_args else {}
                args_by_call_id[it.call_id] = parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                args_by_call_id[it.call_id] = {"raw": raw_args}

    out: list[dict] = []
    for it in items or []:
        if isinstance(it, ToolCallItem) and it.call_id:
            name = name_by_call_id.get(it.call_id, "tool")
            out.append(_tool_start_event(name, args_by_call_id.get(it.call_id, {})))
        elif isinstance(it, ToolCallOutputItem):
            name = name_by_call_id.get(it.call_id or "", "tool")
            out.append(_tool_end_event(name, str(it.output) if it.output is not None else ""))
    return out
