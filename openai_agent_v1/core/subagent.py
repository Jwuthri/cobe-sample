"""Sub-agents as orchestrator tools — one generic wrapper, not N copies.

In the agent-as-tool topology each sub-agent is an SDK ``Agent`` exposed to the
orchestrator as a ``function_tool``. :func:`make_subagent_tool` writes the common
skeleton once and a :class:`SubagentSpec` supplies the per-agent differences as
small plug-in callables.

The skeleton (identical for every sub-agent):
  1. read the shared context off ``wrapper.context``
  2. snapshot domain state (optional, for diffing)
  3. build the sub-agent input from the orchestrator's self-contained ``query``
     (+ optional deterministic state notes) — NOT the chat transcript
  4. run the sub-agent via ``Runner.run_streamed`` and forward its inner tool
     events to the turn's event bus (so the UI sees search_products / add_item …)
  5. tally token usage, distill a ``StepResult``, append it to the context
  6. emit the ``step`` + boundary (router/agent) + deep-trace events
  7. return a TERSE summary string — the only thing the orchestrator LLM reads
     (rich data rides ``StepResult.details`` → deterministic blocks; the model
     can't hallucinate ids/prices it never sees)

**Context isolation.** A sub-agent does NOT see the conversation. The orchestrator
is the sole reader of the transcript: it resolves the user's references ("the green
one", "add it") into a concrete, self-contained ``query`` and passes only that. A
sub-agent operates on ``(query + shared structured state)`` — the cart is its
memory, not the chat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from agents import Agent, RunConfig, Runner, RunContextWrapper, function_tool

from openai_agent_v1.core.config import AgentConfig
from openai_agent_v1.core.context import TurnContext, add_run_usage
from openai_agent_v1.core.factory import agent_max_turns
from openai_agent_v1.core.messages import Msg, human, items_to_msgs, msgs_to_input
from openai_agent_v1.core.step_result import StepResult
from openai_agent_v1.core.trace import emit_trace, render_messages

# Tracing to the OpenAI dashboard is disabled by default for this demo (no project
# wiring); the local deep-trace bus is the observability surface.
_NO_TRACE = RunConfig(tracing_disabled=True)


# ---- plug-in callable signatures (documentation; structural, not enforced) ----
class Extractor(Protocol):
    def __call__(self, ctx: Any, messages: list[Msg], before: Any) -> StepResult: ...


class InputBuilder(Protocol):
    def __call__(self, ctx: Any, query: str) -> list[Msg]: ...


class Snapshot(Protocol):
    def __call__(self, ctx: Any) -> Any: ...


class Summarizer(Protocol):
    def __call__(self, sr: StepResult, ctx: Any) -> str: ...


@dataclass(frozen=True)
class SubagentSpec:
    """A sub-agent definition + the small hooks that make it concrete."""

    name: str  # orchestrator-facing tool name
    description: str  # tool description = the routing surface
    config: AgentConfig | dict  # the declarative agent definition
    extract: Extractor  # REQUIRED: result messages -> StepResult
    build_input: InputBuilder | None = None  # default: just human(query); NO history
    snapshot: Snapshot | None = None  # pre-run snapshot passed to extract as `before`
    summarize: Summarizer | None = None  # default: sr.summary
    block: str | None = None  # writer block kind (consumed by shopping/blocks.py)


def _config_get(config: AgentConfig | dict, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _tool_names(config: AgentConfig | dict) -> list[str]:
    names: list[str] = []
    for spec in _config_get(config, "tools", []) or []:
        name = spec.get("name") if isinstance(spec, dict) else getattr(spec, "name", None)
        if name:
            names.append(name)
    return names


# Sub-agent leaf-tool names that are skills (mapped to a distinct UI row).
_SKILL_TOOLS = {"load_skill"}


async def _run_and_forward(agent: Agent, input_items: list[dict], ctx: Any) -> Any:
    """Run the sub-agent streamed, forwarding inner tool events to the bus.

    Returns the finished ``RunResultStreaming`` (which carries ``new_items`` /
    ``context_wrapper`` once the stream is exhausted).
    """
    result = Runner.run_streamed(
        agent,
        input_items,
        context=ctx,
        max_turns=agent_max_turns(agent),
        run_config=_NO_TRACE,
    )
    name_by_call: dict[str, str] = {}
    async for event in result.stream_events():
        if event.type != "run_item_stream_event":
            continue
        item = event.item
        itype = getattr(item, "type", None)
        raw = getattr(item, "raw_item", None)
        if itype == "tool_call_item":
            name = getattr(raw, "name", None)
            call_id = getattr(raw, "call_id", None)
            if call_id and name:
                name_by_call[call_id] = name
            try:
                args = json.loads(getattr(raw, "arguments", "") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name in _SKILL_TOOLS:
                ctx.emit({"type": "skill", "name": args.get("skill_name")})
            else:
                ctx.emit({"type": "tool_start", "name": name, "args": args})
        elif itype == "tool_call_output_item":
            call_id = raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", None)
            name = name_by_call.get(call_id or "", "")
            if name in _SKILL_TOOLS:
                continue
            content = getattr(item, "output", "")
            content = "" if content is None else str(content)
            if len(content) > 400:
                content = content[:400] + "…"
            ctx.emit({"type": "tool_end", "name": name, "result": content})
    return result


def _step_event(sr: StepResult) -> dict:
    return {
        "type": "step",
        "sop": sr.sop,
        "summary": sr.summary,
        "asks": list(sr.asks),
        "next_sop": sr.next_sop,
        "details": sr.details,
    }


def make_subagent_tool(spec: SubagentSpec, agent: Agent):
    """Wrap a compiled sub-agent as an orchestrator tool, per ``spec``."""

    @function_tool(name_override=spec.name, description_override=spec.description)
    async def _call(wrapper: RunContextWrapper[TurnContext], query: str) -> str:
        """Run the sub-agent on a self-contained instruction.

        Args:
            query: A complete, self-contained instruction for the sub-agent (the
                orchestrator has already resolved any reference into it).
        """
        ctx = wrapper.context
        debug = bool(getattr(ctx, "debug", False))

        # Boundary: routing into this sub-agent (the v4_1 'router' event).
        ctx.emit({"type": "router", "target": spec.name, "iteration": 0})

        before = spec.snapshot(ctx) if spec.snapshot else None

        # Context-isolated: the input is the orchestrator's self-contained query
        # (+ any deterministic state notes the builder adds) — never the transcript.
        if spec.build_input is not None:
            input_msgs: list[Msg] = spec.build_input(ctx, query)
        else:
            input_msgs = [human(query)]

        if debug:  # what the orchestrator sends INTO this sub-agent
            input_seen = render_messages(input_msgs)
            for rendered, raw in zip(input_seen, input_msgs):
                if raw.role == "human" and raw.content == query:
                    rendered["note"] = "orchestrator's instruction (the tool `query`)"
            emit_trace(
                ctx,
                "subagent_input",
                spec.name,
                f"orchestrator → {spec.name}",
                {
                    "query": query,
                    "system_prompt": _config_get(spec.config, "system_prompt", ""),
                    "tools": _tool_names(spec.config),
                    "isolated": True,  # sub-agent does NOT see the conversation
                    "input_seen": input_seen,
                },
            )

        result = await _run_and_forward(agent, msgs_to_input(input_msgs), ctx)
        add_run_usage(ctx.usage, result)

        result_msgs = items_to_msgs(result.new_items)
        sr = spec.extract(ctx, result_msgs, before)
        ctx.step_results.append(sr)
        summary = spec.summarize(sr, ctx) if spec.summarize else sr.summary

        if debug:  # what the sub-agent hands BACK + the mutated runtime context
            emit_trace(
                ctx,
                "subagent_output",
                spec.name,
                f"{spec.name} → orchestrator",
                {
                    "returned_to_orchestrator": summary,
                    "step_result": sr.model_dump(mode="json"),
                    "raw_messages": render_messages(result_msgs),
                },
            )
            emit_trace(
                ctx,
                "context",
                spec.name,
                f"runtime context after {spec.name}",
                ctx.debug_view() if hasattr(ctx, "debug_view") else {},
            )

        # Boundary close (agent) + the distilled step — mirrors v4_1's tool_end order.
        ctx.emit({"type": "agent", "node": f"{spec.name}_wrapper"})
        ctx.emit(_step_event(sr))

        return summary

    return _call


def build_subagent_tools(
    specs: list[SubagentSpec],
    *,
    build_agent: Callable[..., Any],
    context: Any | None = None,
    store: Any | None = None,
    is_enabled: dict[str, Any] | None = None,
) -> list[Any]:
    """Compile each spec's sub-agent and wrap it as a tool — one loop, no copies.

    ``is_enabled`` optionally maps a spec name → an ``is_enabled`` predicate
    (used to hide the checkout delegate while the cart is empty).
    """
    import dataclasses

    tools: list[Any] = []
    for spec in specs:
        agent = build_agent(spec.config, context=context, store=store)
        tool = make_subagent_tool(spec, agent)
        if is_enabled and spec.name in is_enabled:
            tool = dataclasses.replace(tool, is_enabled=is_enabled[spec.name])
        tools.append(tool)
    return tools


__all__ = [
    "SubagentSpec",
    "make_subagent_tool",
    "build_subagent_tools",
]
