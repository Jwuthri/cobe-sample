"""Sub-agents as orchestrator tools — one generic wrapper, not N copies.

In the agent-as-tool topology each sub-agent is a ``create_agent`` exposed to the
orchestrator as a ``@tool``. :func:`as_tool` writes the common skeleton once; a
:class:`SubAgent` supplies the per-agent differences as small plug-in callables.

The skeleton (identical for every sub-agent):
  1. read the shared context off ``runtime.context``;
  2. snapshot domain state (optional, for diffing);
  3. build the sub-agent input from the orchestrator's self-contained ``query``
     (+ optional deterministic state notes) — NOT the chat transcript;
  4. run the sub-agent via :func:`stream_subagent` (re-pumps inner custom events to
     the orchestrator's stream — ``.invoke`` would swallow them);
  5. tally token usage, distill a :class:`~lg_agent.core.step.StepResult`, append it
     to the context;
  6. return a TERSE summary string — the only thing the orchestrator LLM reads.

**Context isolation.** A sub-agent does NOT see the conversation. The orchestrator
is the sole reader of the transcript: it resolves the user's references ("the green
one", "add it") into a concrete, self-contained ``query`` and passes only that. A
sub-agent operates on ``(query + shared structured state)`` — the cart is its
memory, not the chat. This keeps interpretation in one place, cuts tokens +
prompt-injection surface, and makes each sub-agent a clean function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from langchain.tools import ToolRuntime
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from lg_agent.core import trace
from lg_agent.core.config import AgentConfig
from lg_agent.core.context import TurnContext, add_message_usage
from lg_agent.core.step import StepResult


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
class SubAgent:
    """A sub-agent definition + the small hooks that make it concrete.

    The ``config`` is pure JSON (it goes through :class:`AgentConfig`); the hooks
    are the only Python — they translate between the shared domain state and the
    orchestrator. Every field except ``extract`` has a sensible default.
    """

    name: str  # orchestrator-facing tool name (and the StepResult.sop)
    description: str  # tool description = the routing surface the orchestrator reads
    config: AgentConfig | dict  # the declarative agent definition
    extract: Extractor  # REQUIRED: result messages -> StepResult
    build_input: InputBuilder | None = None  # default: just Human(query); NO history
    snapshot: Snapshot | None = None  # pre-run snapshot passed to extract as `before`
    summarize: Summarizer | None = None  # default: sr.summary
    block: str | None = None  # writer block kind (consumed by the writer's blocks)


def stream_subagent(agent: Any, input_state: dict, *, context: Any = None) -> dict:
    """Run a sub-agent via ``stream`` so its custom events reach the parent stream.

    ``.invoke()`` does not propagate a sub-agent's ``get_stream_writer()`` custom
    chunks (tool_start/tool_end/skill) up to the orchestrator, so the UI loses those
    rows. Streaming and re-emitting fixes that. Falls back to ``.invoke()`` if no
    values chunk arrives.
    """
    from langgraph.config import get_stream_writer

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


def as_tool(sub: SubAgent, agent: Any):
    """Wrap a compiled sub-agent as an orchestrator tool, per its :class:`SubAgent`."""

    @tool(sub.name, description=sub.description)
    def _call(query: str, runtime: ToolRuntime[TurnContext] = None) -> str:
        ctx = runtime.context
        debug = bool(getattr(ctx, "debug", False))
        before = sub.snapshot(ctx) if sub.snapshot else None

        # Context-isolated: the input is the orchestrator's self-contained query
        # (+ any deterministic state notes the builder adds) — never the transcript.
        if sub.build_input is not None:
            input_state = sub.build_input(ctx, query)
        else:
            input_state = {"messages": [HumanMessage(content=query)]}

        if debug:
            _trace_input(sub, query, input_state)

        result = stream_subagent(agent, input_state, context=ctx)
        add_message_usage(ctx.usage, result.get("messages", []))

        sr = sub.extract(ctx, result.get("messages", []), before)
        ctx.step_results.append(sr)
        summary = sub.summarize(sr, ctx) if sub.summarize else sr.summary

        if debug:
            _trace_output(sub, ctx, sr, summary, result.get("messages", []))

        return summary

    return _call


def _trace_input(sub: SubAgent, query: str, input_state: dict) -> None:
    """Trace what the orchestrator sends INTO this sub-agent."""
    raw = input_state.get("messages", [])
    input_seen = trace.render_messages(raw)
    # Tag the injected query: it's a 'human' message to the sub-agent, but it is
    # really the ORCHESTRATOR's instruction — not a real user turn.
    for rendered, msg in zip(input_seen, raw):
        if getattr(msg, "type", None) == "human" and getattr(msg, "content", None) == query:
            rendered["note"] = "orchestrator's instruction (the tool `query`)"
    trace.emit(
        "subagent_input",
        sub.name,
        f"orchestrator → {sub.name}",
        {
            "query": query,
            "system_prompt": _config_get(sub.config, "system_prompt", ""),
            "tools": _tool_names(sub.config),
            "isolated": True,  # the sub-agent does NOT see the conversation
            "input_seen": input_seen,
        },
    )


def _trace_output(sub: SubAgent, ctx: Any, sr: StepResult, summary: str, messages: list) -> None:
    """Trace what the sub-agent hands BACK + the mutated runtime context."""
    trace.emit(
        "subagent_output",
        sub.name,
        f"{sub.name} → orchestrator",
        {
            "returned_to_orchestrator": summary,
            "step_result": sr.model_dump(mode="json"),
            "raw_messages": trace.render_messages(messages),
        },
    )
    trace.emit(
        "context",
        sub.name,
        f"runtime context after {sub.name}",
        ctx.debug_view() if hasattr(ctx, "debug_view") else {},
    )


def build_delegate_tools(
    subagents: list[SubAgent],
    *,
    build_agent: Callable[..., Any],
    context_schema: Any | None = None,
    store: Any | None = None,
    checkpointer: Any | None = None,
) -> list[Any]:
    """Compile each sub-agent and wrap it as an orchestrator tool — one loop."""
    tools: list[Any] = []
    for sub in subagents:
        agent = build_agent(
            sub.config,
            context_schema=context_schema,
            store=store,
            checkpointer=checkpointer,
        )
        tools.append(as_tool(sub, agent))
    return tools


__all__ = ["SubAgent", "as_tool", "build_delegate_tools", "stream_subagent"]
