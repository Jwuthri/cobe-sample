"""Regression test for the ``load_skill`` tool's tool_call_id injection.

The bug being prevented: declaring ``tool_call_id: str`` as a regular
function parameter lets the LLM invent its own id (e.g.
``'load_identity_1'``). The ToolMessage we return then doesn't match
the AIMessage's tool_call id, and LangChain's ToolNode raises:

  "Expected to have a matching ToolMessage in Command.update for tool
   'load_skill', got: [ToolMessage(content=..., tool_call_id='load_identity_1')]"

Using ``Annotated[str, InjectedToolCallId]`` removes ``tool_call_id``
from the model-visible schema and injects the real id at runtime.
"""

from __future__ import annotations

from agent_v2.skills import CHECKOUT_SKILLS, make_load_skill_tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command


def test_load_skill_does_not_expose_tool_call_id_to_model():
    """LLM should only see skill_name — not tool_call_id."""
    t = make_load_skill_tool(CHECKOUT_SKILLS)
    assert "skill_name" in t.args
    assert "tool_call_id" not in t.args


def test_load_skill_returns_command_with_matching_tool_call_id():
    """When invoked with a tool_call payload, the returned Command's
    ToolMessage must echo the same tool_call_id."""
    t = make_load_skill_tool(CHECKOUT_SKILLS)
    # Real langchain ToolCall payload shape.
    tool_call = {
        "name": "load_skill",
        "args": {"skill_name": "collect_identity"},
        "id": "call_abc123",
        "type": "tool_call",
    }
    cmd = t.invoke(tool_call)
    assert isinstance(cmd, Command)
    msgs = cmd.update["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], ToolMessage)
    assert msgs[0].tool_call_id == "call_abc123"
    assert "collect_identity" in str(msgs[0].content)
    assert "collect_identity" in cmd.update["skills_loaded"]


def test_load_skill_with_unknown_name_returns_helpful_error():
    t = make_load_skill_tool(CHECKOUT_SKILLS)
    cmd = t.invoke(
        {
            "name": "load_skill",
            "args": {"skill_name": "no_such_skill"},
            "id": "call_zzz",
            "type": "tool_call",
        }
    )
    msg = cmd.update["messages"][0]
    assert msg.tool_call_id == "call_zzz"
    assert "unknown" in str(msg.content).lower()
    # No skills_loaded update on the error path.
    assert "skills_loaded" not in cmd.update
