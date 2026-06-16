"""The SQLite event store + the session's tee-to-DB wrapper."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from lg_agent.core.event_store import SQLiteEventStore
from lg_agent.core.middleware import log_tool_calls
from lg_agent.core.step import StepResult
from lg_agent.shopping.context import ShoppingContext
from lg_agent.shopping.session import ShoppingSession


def test_record_turn_writes_events_and_snapshots(tmp_path):
    store = SQLiteEventStore(str(tmp_path / "ev.db"))
    rows = [
        ("2026-01-01T00:00:00", {"type": "user", "content": "hi", "turn": 1}),
        ("2026-01-01T00:00:01", {"type": "state", "snapshot": {"cart": {"step": "collecting_products"}}}),
        ("2026-01-01T00:00:02", {"type": "bot", "content": "hello", "blocks": []}),
    ]
    store.record_turn("s1", "demo", 1, rows)

    events = store.read_events("s1")
    assert [e["type"] for e in events] == ["user", "state", "bot"]
    assert events[0]["data"]["content"] == "hi"  # full event JSON round-trips
    assert events[0]["ts"] == "2026-01-01T00:00:00"  # per-event timestamp preserved

    snaps = store.read_snapshots("s1")
    assert len(snaps) == 1 and snaps[0]["cart"]["step"] == "collecting_products"
    store.close()


def _fake_session(store):
    from tests_lg.conftest import ToolCallingFake

    @tool("product_rec")
    def _pr(query: str, runtime: ToolRuntime[ShoppingContext] = None) -> str:
        """stub sub-agent"""
        runtime.context.step_results.append(
            StepResult(sop="product_rec", summary="added P-1", details={"added": ["P-1"]}, next_sop="checkout")
        )
        return "added P-1"

    orch = create_agent(
        model=ToolCallingFake(
            messages=iter(
                [
                    AIMessage(content="", tool_calls=[{"name": "product_rec", "args": {"query": "tee"}, "id": "1"}]),
                    AIMessage(content="DONE"),
                ]
            )
        ),
        tools=[_pr],
        system_prompt="route",
        context_schema=ShoppingContext,
        middleware=[log_tool_calls("orch"), ToolCallLimitMiddleware(run_limit=4, exit_behavior="end")],
    )
    writer = create_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="Added the tee.")])),
        tools=[],
        system_prompt="w",
    )
    return ShoppingSession(orchestrator=orch, writer=writer, events_store=store)


def test_session_tees_every_event_to_store(tmp_path):
    store = SQLiteEventStore(str(tmp_path / "ev.db"))
    session = _fake_session(store)
    session.run_turn("add a tee")

    events = store.read_events(session.session_id)
    types = {e["type"] for e in events}
    # main-agent + sub-agent step events all persisted, in stream order
    assert {"user", "state", "router", "step", "bot", "end"} <= types
    assert "trace" in types  # deep-trace frames captured (debug default True)
    # events are ordered (seq increments)
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)
    # at least one state snapshot row landed
    assert len(store.read_snapshots(session.session_id)) >= 1
    store.close()


def test_list_sessions_counts_and_orders(tmp_path):
    store = SQLiteEventStore(str(tmp_path / "ev.db"))
    store.record_turn("s1", "demo", 1, [("2026-01-01T00:00:00", {"type": "user", "content": "a"})])
    store.record_turn("s2", "demo", 1, [("2026-01-02T00:00:00", {"type": "user", "content": "b"})])
    store.record_turn("s2", "demo", 2, [("2026-01-02T00:01:00", {"type": "bot", "content": "c"})])

    sessions = store.list_sessions()
    by_id = {s["session_id"]: s for s in sessions}
    assert by_id["s1"]["turns"] == 1 and by_id["s1"]["events"] == 1
    assert by_id["s2"]["turns"] == 2 and by_id["s2"]["events"] == 2
    assert sessions[0]["session_id"] == "s2"  # most-recent activity first
    store.close()


def test_replayed_events_equal_what_was_streamed(tmp_path):
    # the heart of "looks exactly the same": stored data == the original events,
    # so replaying them is identical to the live stream.
    store = SQLiteEventStore(str(tmp_path / "ev.db"))
    session = _fake_session(store)
    live = session.run_turn("add a tee")["events"]
    replayed = [e["data"] for e in store.read_events(session.session_id)]
    assert replayed == live  # verbatim round-trip
    store.close()


def test_no_store_means_no_writes(tmp_path):
    # a session without a store must run normally and persist nothing
    store = SQLiteEventStore(str(tmp_path / "ev.db"))
    session = _fake_session(store)
    session.events_store = None  # detach
    session.run_turn("add a tee")
    assert store.read_events(session.session_id) == []
    store.close()
