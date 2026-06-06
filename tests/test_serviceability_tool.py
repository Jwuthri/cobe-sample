"""check_serviceability tool — the anonymous lookup for product_rec."""

from __future__ import annotations

from agent_v2.tools.serviceability_tools import check_serviceability


def test_known_zip_returns_options():
    out = check_serviceability.invoke({"zip_code": "94110"})
    assert "Yes" in out or "ship" in out.lower()
    assert "San Francisco" in out
    assert "2h" in out and "next_day" in out


def test_unknown_zip_returns_no():
    out = check_serviceability.invoke({"zip_code": "99999"})
    assert "don't" in out.lower() or "do not" in out.lower()


def test_empty_zip_asks_for_one():
    out = check_serviceability.invoke({"zip_code": ""})
    assert "zip" in out.lower()


def test_serviceability_wrapper_extraction():
    """The graph helper must pull check_serviceability results out of
    the subagent's message history."""
    from agent_v2.graph import _extract_serviceability_from_messages
    from langchain_core.messages import AIMessage, ToolMessage

    msgs = [
        AIMessage(content="checking..."),
        ToolMessage(
            name="check_serviceability",
            content="Yes, we ship to zip 94110 (San Francisco, US). Options: 2h, 4h.",
            tool_call_id="t-1",
        ),
        AIMessage(content="Yes we ship there."),
    ]
    out = _extract_serviceability_from_messages(msgs)
    assert out is not None
    assert "94110" in out["raw"]
    assert "San Francisco" in out["raw"]


def test_serviceability_wrapper_picks_most_recent():
    """If multiple serviceability checks happened, use the last one."""
    from agent_v2.graph import _extract_serviceability_from_messages
    from langchain_core.messages import ToolMessage

    msgs = [
        ToolMessage(name="check_serviceability", content="first zip", tool_call_id="t-1"),
        ToolMessage(name="check_serviceability", content="last zip", tool_call_id="t-2"),
    ]
    out = _extract_serviceability_from_messages(msgs)
    assert out["raw"] == "last zip"


def test_serviceability_wrapper_returns_none_when_absent():
    from agent_v2.graph import _extract_serviceability_from_messages
    from langchain_core.messages import AIMessage, ToolMessage

    msgs = [
        AIMessage(content="hello"),
        ToolMessage(name="search_products", content="P-1: ...", tool_call_id="t-1"),
    ]
    assert _extract_serviceability_from_messages(msgs) is None
