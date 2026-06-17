"""Agent-as-tool — the one wrapper that turns a worker agent into a delegate.

In this topology the orchestrator's "tools" ARE the worker agents. Each delegate tool
the orchestrator can call is just a thin call to :func:`run_subagent`, which writes
the common skeleton once:

  1. announce the routing (``router`` event);
  2. snapshot domain state (optional, for diffing what changed);
  3. run the worker agent on the orchestrator's self-contained ``query`` — passing the
     SHARED ``deps`` as the LangChain ``context`` (one live cart end-to-end);
  4. surface the worker's inner tool calls as ``tool_start`` / ``tool_end`` events
     (parsed from the run's messages — robust + version-stable);
  5. distill a :class:`StepResult`, append it to the turn, emit a ``step`` event;
  6. return a TERSE summary string — the only thing the orchestrator LLM reads back.

**Context isolation.** A worker never sees the conversation. The orchestrator is the
sole reader of the transcript: it resolves references ("the green one", "add it") into
a concrete, self-contained ``query`` and passes only that. A worker operates on
``(query + the shared cart)`` — the cart is its memory, not the chat. This keeps
interpretation in one place, cuts tokens + prompt-injection surface, and makes each
worker a clean function of its inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lg_agent_v3.runtime import events, trace
from lg_agent_v3.runtime.deps import ShoppingDeps
from lg_agent_v3.runtime.step import StepResult

# A worker turns its run's messages (+ a pre-run snapshot) into a StepResult.
Extractor = Callable[[ShoppingDeps, list[Any], Any], StepResult]
Snapshotter = Callable[[ShoppingDeps], Any]
Summarizer = Callable[[StepResult, ShoppingDeps], str]


@dataclass(frozen=True)
class Worker:
    """A worker agent + the small hooks that make it concrete.

    The ``agent`` is a compiled ``create_agent`` graph; the hooks translate between the
    shared domain state and the orchestrator. Only ``extract`` is required.
    """

    name: str  # orchestrator-facing tool name AND the StepResult.sop
    agent: Any  # the compiled LangChain sub-agent
    extract: Extractor  # run messages → StepResult
    prompt: str = ""  # static instructions (shown in the debug trace)
    block: str | None = None  # writer block kind this worker produces
    snapshot: Snapshotter | None = None  # pre-run state, passed to extract as `before`
    summarize: Summarizer | None = None  # default: sr.summary


async def run_subagent(deps: ShoppingDeps, worker: Worker, query: str) -> str:
    """Run one worker on the orchestrator's instruction and report a terse summary."""
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

    hits_before = len(getattr(deps, "guardrail_hits", []))
    result = await worker.agent.ainvoke({"messages": [HumanMessage(content=query)]}, context=deps)
    messages = result.get("messages", [])

    for ev in _tool_events(messages):  # inner tool calls → UI rows
        deps.emit(ev)

    # A guardrail on THIS worker may have fired (input short-circuit or output scrub). A
    # block becomes a flagged step the orchestrator reads as the tool result, and the
    # writer relays it. (A redact just scrubbed the messages — fall through to extract.)
    blocked = next(
        (h for h in deps.guardrail_hits[hits_before:] if h.agent == worker.name and h.action == "block"),
        None,
    )
    if blocked is not None:
        msg = blocked.message or "blocked by a content guardrail"
        sr = StepResult(sop=worker.name, summary=f"[GUARDRAIL] {msg}", details={"guardrail": msg})
        deps.steps.append(sr)
        deps.emit(events.step(sr))
        deps.emit(events.agent(f"{worker.name}_wrapper"))
        return f"GUARDRAIL_BLOCK: {msg}"

    sr = worker.extract(deps, messages, before)
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
                    "raw_messages": trace.render_messages(messages),
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


def tool_returns(messages: list[Any]) -> list[ToolReturn]:
    """Extract ``(tool_name, content)`` pairs from a worker's run messages.

    Workers use this in their ``extract`` hook to read what their tools returned (e.g.
    the catalog lines ``search_products`` produced) without re-deriving state.
    """
    return [ToolReturn(str(m.name), str(m.content)) for m in messages if isinstance(m, ToolMessage)]


def _tool_events(messages: list[Any]) -> list[dict]:
    """Walk a worker's run messages → ordered ``tool_start`` / ``tool_end`` events."""
    out: list[dict] = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                out.append(events.tool_start(tc["name"], _args_dict(tc.get("args"))))
        elif isinstance(m, ToolMessage):
            out.append(events.tool_end(str(m.name), str(m.content)))
    return out


def _args_dict(args: Any) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"raw": args}
    return {}
