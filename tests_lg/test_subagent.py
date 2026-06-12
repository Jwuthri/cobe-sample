"""as_tool: the generic sub-agent wrapper, with a mocked sub-agent."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from lg_agent.core.context import TurnContext
from lg_agent.core.step import StepResult
from lg_agent.core.subagent import SubAgent, as_tool


class _StubAgent:
    def __init__(self, out_messages):
        self.out = out_messages
        self.last_input = None

    def stream(self, input_state, context=None, stream_mode=None):
        self.last_input = input_state
        yield ("values", {"messages": self.out})

    def invoke(self, input_state, context=None):
        self.last_input = input_state
        return {"messages": self.out}


class _FakeRuntime:
    def __init__(self, context, state=None, store=None):
        self.context = context
        self.state = state or {}
        self.store = store


def _sub(**kw):
    base = dict(
        name="x",
        description="d",
        config={"name": "x", "system_prompt": "p"},
        extract=lambda ctx, msgs, before: StepResult(sop="x", summary="did x"),
    )
    base.update(kw)
    return SubAgent(**base)


def test_appends_step_result_and_returns_terse_summary():
    agent = _StubAgent([AIMessage(content="ran")])
    tool = as_tool(_sub(), agent)
    ctx = TurnContext()
    out = tool.func(query="do it", runtime=_FakeRuntime(ctx))
    assert out == "did x"
    assert len(ctx.step_results) == 1 and ctx.step_results[0].summary == "did x"


def test_snapshot_passed_to_extractor():
    seen = {}

    def extract(ctx, msgs, before):
        seen["before"] = before
        return StepResult(sop="x", summary="ok")

    sub = _sub(snapshot=lambda ctx: "SNAP", extract=extract)
    tool = as_tool(sub, _StubAgent([AIMessage(content="ran")]))
    tool.func(query="q", runtime=_FakeRuntime(TurnContext()))
    assert seen["before"] == "SNAP"


def test_default_input_is_query_only():
    # context-isolated: the sub-agent sees ONLY the orchestrator's query — the
    # conversation transcript on runtime.state is deliberately NOT forwarded.
    agent = _StubAgent([AIMessage(content="ran")])
    tool = as_tool(_sub(), agent)
    state = {
        "messages": [
            HumanMessage(content="a"),
            AIMessage(content="b"),
            HumanMessage(content="c"),
        ]
    }
    tool.func(query="Q", runtime=_FakeRuntime(TurnContext(), state=state))
    assert [m.content for m in agent.last_input["messages"]] == ["Q"]


def test_custom_build_input_is_used():
    agent = _StubAgent([AIMessage(content="ran")])
    sub = _sub(build_input=lambda ctx, query: {"messages": [HumanMessage(content=query)]})
    tool = as_tool(sub, agent)
    state = {"messages": [HumanMessage(content="old"), AIMessage(content="older")]}
    tool.func(query="ONLY", runtime=_FakeRuntime(TurnContext(), state=state))
    assert [m.content for m in agent.last_input["messages"]] == ["ONLY"]


def test_custom_summarizer_overrides_summary():
    sub = _sub(summarize=lambda sr, ctx: f"[{sr.summary}!]")
    tool = as_tool(sub, _StubAgent([AIMessage(content="ran")]))
    out = tool.func(query="q", runtime=_FakeRuntime(TurnContext()))
    assert out == "[did x!]"


def test_emits_trace_frames_when_debug(monkeypatch):
    captured: list = []
    monkeypatch.setattr("lg_agent.core.trace.get_stream_writer", lambda: captured.append)

    agent = _StubAgent([AIMessage(content="ran")])
    tool = as_tool(_sub(), agent)
    tool.func(query="do it", runtime=_FakeRuntime(TurnContext(debug=True)))

    traces = [c["trace"] for c in captured if c.get("event") == "trace"]
    assert [t["phase"] for t in traces] == ["subagent_input", "subagent_output", "context"]
    # input frame carries the query + the (isolated) input the sub-agent actually sees
    inp = traces[0]
    assert inp["data"]["query"] == "do it"
    assert inp["data"]["isolated"] is True
    assert [m["content"] for m in inp["data"]["input_seen"]] == ["do it"]  # query only
    # output frame carries the exact terse string handed back to the orchestrator
    assert traces[1]["data"]["returned_to_orchestrator"] == "did x"
    assert traces[1]["data"]["step_result"]["summary"] == "did x"


def test_no_trace_frames_when_debug_off(monkeypatch):
    captured: list = []
    monkeypatch.setattr("lg_agent.core.trace.get_stream_writer", lambda: captured.append)
    tool = as_tool(_sub(), _StubAgent([AIMessage(content="ran")]))
    tool.func(query="q", runtime=_FakeRuntime(TurnContext()))  # debug defaults False
    assert [c for c in captured if c.get("event") == "trace"] == []
