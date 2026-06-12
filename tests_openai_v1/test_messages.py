"""The Msg vocabulary + SDK ↔ Msg conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai_agent_v1.core.messages import ai, human, items_to_msgs, msgs_to_input, system, tool_msg


def test_msgs_to_input_maps_roles_and_drops_tool_and_empty_ai():
    msgs = [human("hi"), ai("hello"), system("memo"), ai(""), tool_msg("t", "out"), human("bye")]
    items = msgs_to_input(msgs)
    assert items == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "memo"},
        {"role": "user", "content": "bye"},
    ]


# --- lightweight fakes mimicking SDK run items ---
@dataclass
class _FakeRaw:
    name: str | None = None
    call_id: str | None = None
    arguments: str | None = None


@dataclass
class _FakeItem:
    type: str
    raw_item: Any
    output: Any = None


def test_items_to_msgs_pairs_calls_with_outputs():
    items = [
        _FakeItem("tool_call_item", _FakeRaw(name="add_item", call_id="c1", arguments='{"product_id":"P-4"}')),
        _FakeItem("tool_call_output_item", {"call_id": "c1"}, output="Added 1 × Cap."),
    ]
    msgs = items_to_msgs(items)
    assert msgs[0].role == "ai" and msgs[0].tool_calls[0]["name"] == "add_item"
    assert msgs[1].role == "tool"
    assert msgs[1].name == "add_item"
    assert msgs[1].content == "Added 1 × Cap."


def test_items_to_msgs_handles_dict_output_raw_item():
    items = [
        _FakeItem("tool_call_item", _FakeRaw(name="search_products", call_id="c2", arguments="{}")),
        _FakeItem("tool_call_output_item", {"call_id": "c2", "output": "P-2: Black Hoodie"}, output=None),
    ]
    msgs = items_to_msgs(items)
    tool = [m for m in msgs if m.role == "tool"][0]
    assert tool.name == "search_products"
    assert "Black Hoodie" in tool.content
