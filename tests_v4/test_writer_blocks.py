"""writer() returns the {message, blocks} envelope (draft_response + draft_blocks)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent_v4 import ids
from agent_v4.state import AgentState
from agent_v4.step_result import StepResult
from agent_v4.writer import writer
from langchain_core.messages import HumanMessage
from langgraph.types import Command


class FakeChat:
    def __init__(self, *_, **__) -> None:
        pass

    def invoke(self, _messages):
        return MagicMock(content="Here are some options:")


def test_writer_returns_message_and_blocks_for_product_rec():
    s = AgentState(
        user_id="u",
        session_id="s",
        messages=[HumanMessage(content="show me hoodies")],
        step_results=[
            StepResult(
                sop=ids.PRODUCT_REC,
                details={"products": [{"id": "P-2", "name": "Black Hoodie", "price": "49.99", "tags": []}]},
            )
        ],
    )
    with patch("agent_v4.writer.ChatOpenAI", FakeChat):
        cmd = writer(s)
    assert isinstance(cmd, Command)
    assert cmd.goto == "validator"
    assert cmd.update["draft_response"] == "Here are some options:"
    blocks = cmd.update["draft_blocks"]
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "product_reco"
    assert blocks[0]["products"][0]["id"] == "P-2"


def test_writer_returns_empty_blocks_for_smalltalk():
    s = AgentState(user_id="u", session_id="s", messages=[HumanMessage(content="hi")])
    with patch("agent_v4.writer.ChatOpenAI", FakeChat):
        cmd = writer(s)
    assert cmd.update["draft_blocks"] == []
    assert cmd.update["draft_response"]  # prose still produced
