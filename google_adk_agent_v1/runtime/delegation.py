"""Agent-as-tool — the one wrapper that turns a worker agent into a delegate.

In this topology the orchestrator's "tools" ARE the worker agents. Each delegate
tool the orchestrator can call is just a thin call to :func:`run_subagent`, which
writes the common skeleton once:

  1. announce the routing (``router`` event);
  2. snapshot domain state (optional, for diffing what changed);
  3. run the worker agent on the orchestrator's self-contained ``query`` — in a fresh,
     isolated ADK session that shares the SAME live cart (via the registry key copied
     out of the orchestrator's session state);
  4. surface the worker's inner tool calls as ``tool_start`` / ``tool_end`` events;
  5. distill a :class:`StepResult`, append it to the turn, emit a ``step`` event;
  6. return a TERSE summary string — the only thing the orchestrator LLM reads back.

**Context isolation.** A worker never sees the conversation. The orchestrator is the
sole reader of the transcript: it resolves references ("the green one", "add it")
into a concrete, self-contained ``query`` and passes only that. A worker operates on
``(query + the shared cart)`` — the cart is its memory, not the chat.

**One worker at a time.** ``deps.lock`` serializes delegated runs, so even a compound
user message ("a green cap AND check ORD-7") mutates the one cart and feeds the one
event bus sequentially — ADK's analogue of Pydantic AI's ``parallel_tool_calls=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

from google_adk_agent_v1.runtime import events, trace
from google_adk_agent_v1.runtime import registry
from google_adk_agent_v1.runtime.deps import ShoppingDeps
from google_adk_agent_v1.runtime.runner_util import run_collect
from google_adk_agent_v1.runtime.step import StepResult

# A worker turns its run's events (+ a pre-run snapshot) into a StepResult.
Extractor = Callable[[ShoppingDeps, list[Any], Any], StepResult]
Snapshotter = Callable[[ShoppingDeps], Any]
Summarizer = Callable[[StepResult, ShoppingDeps], str]


@dataclass(frozen=True)
class Worker:
    """A worker agent + the small hooks that make it concrete.

    The ``agent`` is a plain ADK ``LlmAgent``; the hooks translate between the shared
    domain state and the orchestrator. Only ``extract`` is required.
    """

    name: str  # orchestrator-facing tool name AND the StepResult.sop
    agent: LlmAgent  # the ADK sub-agent
    extract: Extractor  # run events → StepResult
    prompt: str = ""  # static instructions (shown in the debug trace)
    block: str | None = None  # writer block kind this worker produces
    snapshot: Snapshotter | None = None  # pre-run state, passed to extract as `before`
    summarize: Summarizer | None = None  # default: sr.summary


def deps_from(tool_context: ToolContext) -> ShoppingDeps:
    """Resolve the one live ``ShoppingDeps`` from the registry key in session state."""
    return registry.get(tool_context.state[registry.RUNTIME_KEY])


async def run_subagent(tool_context: ToolContext, worker: Worker, query: str) -> str:
    """Run one worker on the orchestrator's instruction and report a terse summary."""
    key = tool_context.state[registry.RUNTIME_KEY]
    deps = registry.get(key)

    # Serialize delegated runs: one worker mutates the cart + drains onto the bus at a
    # time, so a compound message stays ordered and race-free.
    async with deps.lock:
        deps.emit(events.router(worker.name))
        before = worker.snapshot(deps) if worker.snapshot else None

        if deps.debug:
            deps.emit(
                trace.frame(
                    "subagent_input",
                    worker.name,
                    f"orchestrator → {worker.name}",
                    {
                        "query": query,
                        "system_prompt": worker.prompt,
                        "isolated": True,  # the worker does NOT see the conversation
                        "input_seen": [{"role": "human", "content": query, "note": "orchestrator's instruction"}],
                    },
                )
            )

        run_events = await run_collect(worker.agent, text=query, runtime_key=key)

        for ev in _tool_events(run_events):  # inner tool calls → UI rows
            deps.emit(ev)

        sr = worker.extract(deps, run_events, before)
        deps.steps.append(sr)
        deps.emit(events.step(sr))
        summary = worker.summarize(sr, deps) if worker.summarize else sr.summary

        if deps.debug:
            deps.emit(
                trace.frame(
                    "subagent_output",
                    worker.name,
                    f"{worker.name} → orchestrator",
                    {
                        "returned_to_orchestrator": summary,
                        "step_result": sr.model_dump(mode="json"),
                        "raw_messages": trace.render_messages(run_events),
                    },
                )
            )
            deps.emit(trace.frame("context", worker.name, f"context after {worker.name}", deps.debug_view()))

        deps.emit(events.agent(f"{worker.name}_wrapper"))
        return summary


@dataclass(frozen=True)
class ToolReturn:
    name: str
    content: str


def _unwrap(response: Any) -> Any:
    """ADK wraps a tool's non-dict return as ``{"result": <value>}`` — unwrap it."""
    if isinstance(response, dict) and set(response) == {"result"}:
        return response["result"]
    return response


def tool_returns(events_list: list[Any]) -> list[ToolReturn]:
    """Extract ``(tool_name, content)`` pairs from a worker's run events.

    Workers use this in their ``extract`` hook to read what their tools returned
    (e.g. the catalog lines ``search_products`` produced) without re-deriving state.
    """
    out: list[ToolReturn] = []
    for ev in events_list:
        for fr in ev.get_function_responses():
            out.append(ToolReturn(fr.name, str(_unwrap(fr.response))))
    return out


def _tool_events(events_list: list[Any]) -> list[dict]:
    """Walk a worker's run events → ordered ``tool_start`` / ``tool_end`` events.

    Parts are visited in emission order: a ``function_call`` part is a tool start, a
    ``function_response`` part is the matching end.
    """
    out: list[dict] = []
    for ev in events_list:
        content = getattr(ev, "content", None)
        if not content or not content.parts:
            continue
        for part in content.parts:
            fc = getattr(part, "function_call", None)
            if fc is not None:
                out.append(events.tool_start(fc.name, dict(fc.args or {})))
            fr = getattr(part, "function_response", None)
            if fr is not None:
                out.append(events.tool_end(fr.name, str(_unwrap(fr.response))))
    return out
