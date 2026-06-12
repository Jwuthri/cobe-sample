"""Sub-agents as orchestrator tools — one generic wrapper, not N copies.

In the agent-as-tool topology each sub-agent is an ``agno.Agent`` exposed to the
orchestrator as a tool. :func:`make_subagent_tool` writes the common skeleton once
and a :class:`SubagentSpec` supplies the per-agent differences as small plug-in
callables (the :mod:`~agno_agent_v1.agent.extractors`).

The skeleton (identical for every sub-agent):
  1. read the shared :class:`ShoppingContext` off ``run_context.dependencies``
  2. snapshot domain state (optional, for diffing)
  3. build the sub-agent input from the orchestrator's self-contained ``query``
     (+ optional deterministic state notes) — NOT the chat transcript
  4. run the sub-agent (propagating the same ``dependencies`` so its tools mutate
     the one shared cart)
  5. distill a ``StepResult`` from the run's tool executions, append it to ctx
  6. push UI events (router / tool / step / trace) onto ``ctx.events`` and return
     a TERSE summary string — the only thing the orchestrator LLM ever reads

**Context isolation.** A sub-agent does NOT see the conversation. The orchestrator
is the sole reader of the transcript: it resolves the user's references ("the
green one", "add it") into a concrete, self-contained ``query`` and passes only
that. A sub-agent operates on ``(query + shared structured state)`` — the cart is
its memory, not the chat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agno.models.message import Message
from agno.run import RunContext
from agno.tools.function import Function

from agno_agent_v1.agent.builder import AgentSpec
from agno_agent_v1.agent.context import ShoppingContext
from agno_agent_v1.agent.events import render_messages, step_event, trace_event
from agno_agent_v1.agent.extractors import _tool_name, _tool_result


@dataclass(frozen=True)
class SubagentSpec:
    """A sub-agent definition + the small hooks that make it concrete."""

    name: str  # orchestrator-facing tool name
    description: str  # tool description = the routing surface
    config: AgentSpec  # the declarative agent definition
    extract: Callable[[ShoppingContext, list[Any], Any], Any]  # tool execs -> StepResult
    build_input: Callable[[ShoppingContext, str], list[Message]] | None = None
    snapshot: Callable[[ShoppingContext], Any] | None = None  # pre-run snapshot ("before")
    summarize: Callable[[Any, ShoppingContext], str] | None = None
    block: str | None = None  # writer block kind (consumed by blocks.py)


def _add_run_usage(usage: dict[str, int], run_output: Any) -> None:
    """Tally token usage from an Agno RunOutput's metrics (best-effort)."""
    metrics = getattr(run_output, "metrics", None)
    if metrics is None:
        return
    usage["input_tokens"] += int(getattr(metrics, "input_tokens", 0) or 0)
    usage["output_tokens"] += int(getattr(metrics, "output_tokens", 0) or 0)
    usage["llm_calls"] += 1


def _tool_event_pairs(tool_calls: list[Any]) -> list[dict]:
    """Reconstruct tool_start / tool_end UI events from a run's tool executions."""
    events: list[dict] = []
    for tc in tool_calls:
        name = _tool_name(tc)
        args = {k: v for k, v in (getattr(tc, "tool_args", None) or {}).items()}
        events.append({"type": "tool_start", "name": name, "args": args})
        events.append({"type": "tool_end", "name": name, "result": _tool_result(tc)})
    return events


def make_subagent_tool(spec: SubagentSpec, agent: Any) -> Function:
    """Wrap a compiled sub-agent as an orchestrator tool, per ``spec``."""

    def _call(query: str, run_context: RunContext) -> str:
        ctx: ShoppingContext = (run_context.dependencies or {})["ctx"]
        debug = bool(getattr(ctx, "debug", False))
        before = spec.snapshot(ctx) if spec.snapshot else None

        # Context-isolated: the input is the orchestrator's self-contained query
        # (+ any deterministic state notes the builder adds) — never the transcript.
        input_messages = (
            spec.build_input(ctx, query)
            if spec.build_input is not None
            else [Message(role="user", content=query)]
        )

        if debug:  # what the orchestrator sends INTO this sub-agent
            input_seen = render_messages(input_messages)
            for rendered, raw in zip(input_seen, input_messages):
                if getattr(raw, "role", None) == "user" and getattr(raw, "content", None) == query:
                    rendered["note"] = "orchestrator's instruction (the tool `query`)"
            ctx.events.append(
                trace_event(
                    "subagent_input",
                    spec.name,
                    f"orchestrator → {spec.name}",
                    {
                        "query": query,
                        "system_prompt": spec.config.prompt,
                        "tools": [getattr(t, "__name__", str(t)) for t in spec.config.tools],
                        "isolated": True,  # sub-agent does NOT see the conversation
                        "input_seen": input_seen,
                    },
                )
            )

        # router row: the orchestrator is delegating to this sub-agent
        ctx.events.append({"type": "router", "target": spec.name, "iteration": 0})

        # run the sub-agent, propagating the shared dependencies (one live cart)
        run_output = agent.run(input_messages, dependencies=run_context.dependencies)
        tool_calls = list(getattr(run_output, "tools", None) or [])
        _add_run_usage(ctx.usage, run_output)

        # surface the inner tool calls, then mark the wrapper done
        ctx.events.extend(_tool_event_pairs(tool_calls))
        ctx.events.append({"type": "agent", "node": f"{spec.name}_wrapper"})

        # distill the StepResult and append it to the shared turn context
        sr = spec.extract(ctx, tool_calls, before)
        ctx.step_results.append(sr)
        summary = spec.summarize(sr, ctx) if spec.summarize else sr.summary

        if debug:  # what the sub-agent hands BACK + the mutated runtime context
            ctx.events.append(
                trace_event(
                    "subagent_output",
                    spec.name,
                    f"{spec.name} → orchestrator",
                    {
                        "returned_to_orchestrator": summary,
                        "step_result": sr.model_dump(mode="json"),
                        "tool_calls": [
                            {"name": _tool_name(tc), "result": _tool_result(tc)} for tc in tool_calls
                        ],
                    },
                )
            )
            ctx.events.append(
                trace_event("context", spec.name, f"runtime context after {spec.name}", ctx.debug_view())
            )

        # the step row carries the structured result to the UI
        ctx.events.append(step_event(sr))
        return summary

    _call.__name__ = spec.name
    _call.__doc__ = spec.description
    fn = Function.from_callable(_call)
    fn.description = spec.description
    return fn


def build_subagent_tools(
    specs: list[SubagentSpec], *, build_agent: Callable[[AgentSpec], Any]
) -> dict[str, Function]:
    """Compile each spec's sub-agent and wrap it as a tool — keyed by name."""
    return {spec.name: make_subagent_tool(spec, build_agent(spec.config)) for spec in specs}


__all__ = ["SubagentSpec", "make_subagent_tool", "build_subagent_tools"]
