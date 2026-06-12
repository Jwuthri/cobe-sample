"""Sub-agents as orchestrator tools — one generic wrapper, not N copies.

In the agent-as-tool topology each sub-agent is a ``create_agent`` exposed to the
orchestrator as a ``@tool``. v5 hand-wrote three ~90-line wrappers that differed
in only a few spots; here :func:`make_subagent_tool` writes the common skeleton
once and a :class:`SubagentSpec` supplies the per-agent differences as small
plug-in callables.

The skeleton (identical for every sub-agent):
  1. read the shared context off ``runtime.context``
  2. snapshot domain state (optional, for diffing)
  3. build the sub-agent input from the orchestrator's self-contained ``query``
     (+ optional deterministic state notes) — NOT the chat transcript
  4. run the sub-agent via :func:`stream_subagent` (re-pumps inner custom events
     to the orchestrator's stream — ``.invoke`` would swallow them)
  5. tally token usage, distill a ``StepResult``, append it to the context
  6. return a TERSE summary string — the only thing the orchestrator LLM reads
     (rich data rides ``StepResult.details`` → deterministic blocks; the model
     can't hallucinate ids/prices it never sees)

**Context isolation.** A sub-agent does NOT see the conversation. The orchestrator
is the sole reader of the transcript: it resolves the user's references ("the green
one", "add it") into a concrete, self-contained ``query`` and passes only that. A
sub-agent operates on ``(query + shared structured state)`` — the cart is its
memory, not the chat. This keeps responsibility for interpretation in one place,
cuts tokens + prompt-injection surface, and makes each sub-agent a clean function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from langchain.tools import ToolRuntime
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer

from agent_v4_1.core.config import AgentConfig
from agent_v4_1.core.context import TurnContext, add_message_usage
from agent_v4_1.core.step_result import StepResult
from agent_v4_1.core.trace import emit_trace, render_messages


# ---- plug-in callable signatures (documentation; structural, not enforced) ----
class Extractor(Protocol):
    def __call__(self, ctx: Any, messages: list[BaseMessage], before: Any) -> StepResult: ...


class InputBuilder(Protocol):
    def __call__(self, ctx: Any, query: str) -> dict: ...


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
    build_input: InputBuilder | None = None  # default: just Human(query); NO history
    snapshot: Snapshot | None = None  # pre-run snapshot passed to extract as `before`
    summarize: Summarizer | None = None  # default: sr.summary
    block: str | None = None  # writer block kind (consumed by shopping/blocks.py)


def stream_subagent(agent: Any, input_state: dict, *, context: Any = None) -> dict:
    """Run a sub-agent via ``stream`` so its custom events reach the parent stream.

    ``.invoke()`` does not propagate a sub-agent's ``get_stream_writer()`` custom
    chunks (tool_start/tool_end/skill) up to the orchestrator, so the UI loses
    those rows. Streaming and re-emitting fixes that. Falls back to ``.invoke()``
    if no values chunk arrives.
    """
    writer = None
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None

    result: dict | None = None
    for chunk in agent.stream(input_state, context=context, stream_mode=["custom", "values"]):
        if isinstance(chunk, tuple) and len(chunk) == 2:
            mode, payload = chunk
        else:
            mode, payload = "values", chunk
        if mode == "custom" and writer is not None and isinstance(payload, dict):
            writer(payload)
        elif mode == "values" and isinstance(payload, dict):
            result = payload

    if result is not None:
        return result
    return agent.invoke(input_state, context=context)


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


def make_subagent_tool(spec: SubagentSpec, agent: Any):
    """Wrap a compiled sub-agent as an orchestrator tool, per ``spec``."""

    @tool(spec.name, description=spec.description)
    def _call(query: str, runtime: ToolRuntime[TurnContext] = None) -> str:
        ctx = runtime.context
        debug = bool(getattr(ctx, "debug", False))
        before = spec.snapshot(ctx) if spec.snapshot else None

        # Context-isolated: the input is the orchestrator's self-contained query
        # (+ any deterministic state notes the builder adds) — never the transcript.
        if spec.build_input is not None:
            input_state = spec.build_input(ctx, query)
        else:
            input_state = {"messages": [HumanMessage(content=query)]}

        if debug:  # what the orchestrator sends INTO this sub-agent
            raw_input_msgs = input_state.get("messages", [])
            input_seen = render_messages(raw_input_msgs)
            # tag the injected query: it's a 'human' message to the sub-agent, but
            # it's really the ORCHESTRATOR's instruction — not a real user turn.
            for rendered, raw in zip(input_seen, raw_input_msgs):
                if getattr(raw, "type", None) == "human" and getattr(raw, "content", None) == query:
                    rendered["note"] = "orchestrator's instruction (the tool `query`)"
            emit_trace(
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

        result = stream_subagent(agent, input_state, context=ctx)
        add_message_usage(ctx.usage, result.get("messages", []))

        sr = spec.extract(ctx, result.get("messages", []), before)
        ctx.step_results.append(sr)
        summary = spec.summarize(sr, ctx) if spec.summarize else sr.summary

        if debug:  # what the sub-agent hands BACK + the mutated runtime context
            emit_trace(
                "subagent_output",
                spec.name,
                f"{spec.name} → orchestrator",
                {
                    "returned_to_orchestrator": summary,
                    "step_result": sr.model_dump(mode="json"),
                    "raw_messages": render_messages(result.get("messages", [])),
                },
            )
            emit_trace(
                "context",
                spec.name,
                f"runtime context after {spec.name}",
                ctx.debug_view() if hasattr(ctx, "debug_view") else {},
            )

        return summary

    return _call


def build_subagent_tools(
    specs: list[SubagentSpec],
    *,
    build_agent: Callable[..., Any],
    context_schema: Any | None = None,
    store: Any | None = None,
    checkpointer: Any | None = None,
) -> list[Any]:
    """Compile each spec's sub-agent and wrap it as a tool — one loop, no copies."""
    tools: list[Any] = []
    for spec in specs:
        agent = build_agent(
            spec.config,
            context_schema=context_schema,
            store=store,
            checkpointer=checkpointer,
        )
        tools.append(make_subagent_tool(spec, agent))
    return tools


__all__ = [
    "SubagentSpec",
    "make_subagent_tool",
    "build_subagent_tools",
    "stream_subagent",
]
