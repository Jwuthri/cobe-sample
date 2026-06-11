"""make_subagent_tool: the generic wrapper, with a mocked sub-agent."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent_v4_1.core.context import TurnContext
from agent_v4_1.core.step_result import StepResult
from agent_v4_1.core.subagent import SubagentSpec, make_subagent_tool


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


def _spec(**kw):
    base = dict(
        name="x",
        description="d",
        config={"name": "x", "system_prompt": "p"},
        extract=lambda ctx, msgs, before: StepResult(sop="x", summary="did x"),
    )
    base.update(kw)
    return SubagentSpec(**base)


def test_appends_step_result_and_returns_terse_summary():
    agent = _StubAgent([AIMessage(content="ran")])
    tool = make_subagent_tool(_spec(), agent)
    ctx = TurnContext()
    out = tool.func(query="do it", runtime=_FakeRuntime(ctx))
    assert out == "did x"
    assert len(ctx.step_results) == 1 and ctx.step_results[0].summary == "did x"


def test_snapshot_passed_to_extractor():
    seen = {}

    def extract(ctx, msgs, before):
        seen["before"] = before
        return StepResult(sop="x", summary="ok")

    spec = _spec(snapshot=lambda ctx: "SNAP", extract=extract)
    tool = make_subagent_tool(spec, _StubAgent([AIMessage(content="ran")]))
    tool.func(query="q", runtime=_FakeRuntime(TurnContext()))
    assert seen["before"] == "SNAP"


def test_default_input_windows_clean_history():
    agent = _StubAgent([AIMessage(content="ran")])
    tool = make_subagent_tool(_spec(history_window=2), agent)
    state = {
        "messages": [
            HumanMessage(content="a"),
            AIMessage(content="b"),
            HumanMessage(content="c"),
            AIMessage(content="d"),
        ]
    }
    tool.func(query="Q", runtime=_FakeRuntime(TurnContext(), state=state))
    contents = [m.content for m in agent.last_input["messages"]]
    assert contents == ["c", "d", "Q"]  # last 2 history msgs + the query


def test_custom_build_input_is_used():
    agent = _StubAgent([AIMessage(content="ran")])
    spec = _spec(build_input=lambda ctx, history, query: {"messages": [HumanMessage(content=query)]})
    tool = make_subagent_tool(spec, agent)
    state = {"messages": [HumanMessage(content="old"), AIMessage(content="older")]}
    tool.func(query="ONLY", runtime=_FakeRuntime(TurnContext(), state=state))
    assert [m.content for m in agent.last_input["messages"]] == ["ONLY"]


def test_custom_summarizer_overrides_summary():
    spec = _spec(summarize=lambda sr, ctx: f"[{sr.summary}!]")
    tool = make_subagent_tool(spec, _StubAgent([AIMessage(content="ran")]))
    out = tool.func(query="q", runtime=_FakeRuntime(TurnContext()))
    assert out == "[did x!]"
