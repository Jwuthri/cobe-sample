"""The Agno-stream → SSE event bridge helpers."""

from __future__ import annotations

from agent_agno_v1.core import events as E
from agent_agno_v1.core.step_result import StepResult
from tests_agno_v1.fakes import FakeEvent, FakeTool


def test_event_name_and_content_delta():
    ev = FakeEvent("TeamRunContent", content="hi", content_type="str")
    assert E.ev_name(ev) == "TeamRunContent"
    assert E.ev_content(ev) == "hi"
    # non-string content (e.g. structured-output object) yields no token text
    assert E.ev_content(FakeEvent("TeamRunContent", content={"x": 1}, content_type="json")) == ""


def test_tool_accessors_and_sse_shapes():
    tool = FakeTool("search_products", {"query": "h"}, result="P-2: ...")
    assert E.tool_name(tool) == "search_products"
    assert E.tool_args(tool) == {"query": "h"}
    assert E.tool_start_event(tool) == {"type": "tool_start", "name": "search_products", "args": {"query": "h"}}
    assert E.tool_end_event(tool)["type"] == "tool_end"


def test_delegate_target_and_canonicalisation():
    tool = FakeTool("delegate_task_to_member", {"member_id": "order-status", "task": "x"})
    assert E.delegate_target(tool) == "order-status"
    assert E.canonical_member("order-status") == "order_status"
    assert E.canonical_member("product-rec") == "product_rec"
    assert E.canonical_member("checkout") == "checkout"


def test_member_not_found_detection():
    assert E.is_member_not_found("Member with ID product_rec not found in the team")
    assert not E.is_member_not_found("ok")


def test_router_and_step_events():
    assert E.router_event("product_rec") == {"type": "router", "target": "product_rec", "iteration": 0}
    sr = StepResult(sop="checkout", summary="did stuff", asks=["name"], next_sop=None, details={"x": 1})
    se = E.step_event(sr)
    assert se == {
        "type": "step",
        "sop": "checkout",
        "summary": "did stuff",
        "asks": ["name"],
        "next_sop": None,
        "details": {"x": 1},
    }
