"""The streaming session pipeline, driven by a fake team + fake writer (no LLM).

Verifies the full event vocabulary, StepResult extraction from member responses,
deterministic block assembly, live token streaming, and the snapshot state.
"""

from __future__ import annotations

from agent_agno_v4_1.session import ShoppingSession
from tests_agno_v4_1.conftest import FakeTeam, FakeWriter, member_response, tool_exec


def _session(script, deltas) -> ShoppingSession:
    return ShoppingSession(team=FakeTeam(script), writer=FakeWriter(deltas))


def _types(events) -> list[str]:
    return [e["type"] for e in events]


def test_product_rec_turn_full_pipeline():
    def script(ctx):
        ctx.cart_service.add_item("P-1", 2)  # what the live member would do
        return [member_response("product_rec", [tool_exec("add_item", "Added 2 × Tee")])]

    s = _session(script, ["Added ", "2 of P-1."])
    out = s.run_turn("add 2 of P-1")

    # event vocabulary
    t = _types(out["events"])
    for expected in ("user", "router", "tool_start", "tool_end", "agent", "step", "token", "bot", "end"):
        assert expected in t, f"missing {expected} in {t}"

    # routing + extraction
    routers = [e["target"] for e in out["events"] if e["type"] == "router"]
    assert routers == ["product_rec", "writer"]
    step = next(e for e in out["events"] if e["type"] == "step")
    assert step["sop"] == "product_rec"
    assert step["next_sop"] == "checkout"

    # deterministic block + streamed text + cart mutation
    assert [b["kind"] for b in out["blocks"]] == ["product_reco"]
    assert out["blocks"][0]["added_ids"] == ["P-1"]
    assert out["tokens"] == ["Added ", "2 of P-1."]
    assert out["message"] == "Added 2 of P-1."
    assert [(i.product_id, i.quantity) for i in s.cart_service.cart.items] == [("P-1", 2)]


def test_smalltalk_turn_has_no_members_no_blocks():
    s = _session(lambda ctx: [], ["Hi there!"])
    out = s.run_turn("hello")
    assert out["blocks"] == []
    assert out["message"] == "Hi there!"
    # no member routing, only the writer router event
    assert [e["target"] for e in out["events"] if e["type"] == "router"] == ["writer"]


def test_empty_writer_falls_back():
    s = _session(lambda ctx: [], [])  # writer yields nothing twice
    out = s.run_turn("hello")
    assert out["message"].startswith("Sorry, I couldn't produce a response")


def test_quantity_edit_drives_awaiting_pricing_snapshot():
    """Re-pricing carries over: editing a ready cart blocks confirm until recompute."""

    def setup(ctx):
        c = ctx.cart_service
        c.add_item("P-1", 2)
        c.set_customer("J", "W")
        c.set_address("1 Main", "SF", "94110")
        c.lookup_serviceability()
        c.set_delivery_option("standard")
        c.quote_shipping()
        c.compute_tax()
        c.attach_payment("cash")
        return [member_response("checkout", [tool_exec("attach_payment", "ok")])]

    s = _session(setup, ["Ready to confirm."])
    s.run_turn("checkout with standard + cash")
    assert s.cart_service.cart.step.value == "ready_to_confirm"

    # now decrease quantity -> shipping/tax stale -> awaiting_pricing
    def decrease(ctx):
        ctx.cart_service.set_quantity("P-1", 1)
        return [member_response("product_rec", [tool_exec("set_quantity", "Set P-1 to 1")])]

    s.team = FakeTeam(decrease)
    s.writer = FakeWriter(["Updated to 1."])
    out = s.run_turn("make it 1")
    snap = next(e["snapshot"] for e in reversed(out["events"]) if e["type"] == "state")
    assert snap["cart"]["step"] == "awaiting_pricing"
    assert snap["cart"]["grand_total"] is None
