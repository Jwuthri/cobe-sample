"""emit carries blocks via AIMessage.additional_kwargs; serialize_state surfaces them."""

from __future__ import annotations

from agent_v4.graph import emit
from agent_v4.state import AgentState
from langchain_core.messages import AIMessage
from langgraph.types import Command


def _block() -> dict:
    return {
        "kind": "product_reco",
        "products": [{"id": "P-2", "name": "Black Hoodie", "price": "49.99", "tags": []}],
        "added_ids": [],
        "serviceability": None,
    }


def test_emit_attaches_blocks_to_additional_kwargs():
    s = AgentState(
        user_id="u",
        session_id="s",
        draft_response="Here are some options:",
        draft_blocks=[_block()],
    )
    cmd = emit(s)
    assert isinstance(cmd, Command)
    msg = cmd.update["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.additional_kwargs == {"blocks": [_block()]}
    # blocks are cleared from state after emit (don't leak across turns)
    assert cmd.update["draft_blocks"] == []


def test_emit_no_additional_kwargs_when_no_blocks():
    s = AgentState(user_id="u", session_id="s", draft_response="Hi there!", draft_blocks=[])
    cmd = emit(s)
    msg = cmd.update["messages"][0]
    assert msg.additional_kwargs == {}


def test_serialize_state_surfaces_message_blocks():
    from server.main_v4 import serialize_state

    s = AgentState(
        user_id="u",
        session_id="s",
        messages=[AIMessage(content="Here are some options:", additional_kwargs={"blocks": [_block()]})],
    )
    snap = serialize_state(s)
    last = snap["messages"][-1]
    assert last["content"] == "Here are some options:"
    assert last["blocks"] == [_block()]
