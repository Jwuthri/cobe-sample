"""End-to-end pipeline tests driven by scripted FunctionModels (no real LLM).

These exercise the whole streaming turn — guardrails → orchestrator (delegation) →
worker tools → deterministic blocks → streamed writer — and the wire events the UI
consumes, all offline and deterministically.
"""

from __future__ import annotations

from tests_pydantic_agent_v1.conftest import call_then_done, say, sequence

from pydantic_agent_v1 import ShoppingSession


def _kinds(events, kind):
    return [e for e in events if e["type"] == kind]


def test_browse_and_add_routes_to_product_rec(override):
    s = ShoppingSession()
    with override(
        orchestrator=call_then_done(("product_rec", {"query": "add P-2 to the cart"})),
        product_rec=call_then_done(("add_item", {"product_id": "P-2"})),
        writer=say("Added the Black Hoodie to your cart."),
    ):
        out = s.run_turn("add the black hoodie")
    events = out["events"]

    # routing + tool + step events surfaced
    assert [r["target"] for r in _kinds(events, "router")] == ["product_rec", "writer"]
    assert any(t["name"] == "add_item" for t in _kinds(events, "tool_start"))
    assert _kinds(events, "step")[0]["sop"] == "product_rec"

    # the cart actually mutated, and the reply streamed token-by-token
    assert [i["id"] for i in s.snapshot()["cart"]["items"]] == ["P-2"]
    assert out["tokens"], "writer should have streamed at least one token"
    assert out["message"] == "Added the Black Hoodie to your cart."

    # a deterministic product_reco block was attached
    assert any(b["kind"] == "product_reco" for b in out["blocks"])


def test_smalltalk_calls_no_worker(override):
    s = ShoppingSession()
    with override(
        orchestrator=say("DONE"),  # routes nothing
        writer=say("Hi! I can help you find products and place orders."),
    ):
        out = s.run_turn("hello there")
    assert [r["target"] for r in _kinds(out["events"], "router")] == ["writer"]
    assert not _kinds(out["events"], "step")
    assert out["blocks"] == []
    assert out["message"].startswith("Hi!")


def test_checkout_captures_address_and_advances(override):
    s = ShoppingSession()
    s.cart_service.add_item("P-2", 1)
    s.cart_service.set_customer("Ada", "Lovelace")  # now at collecting_address

    with override(
        orchestrator=call_then_done(("checkout", {"query": "1 Market St, San Francisco, 94105"})),
        checkout=sequence(
            ("set_address", {"street": "1 Market St", "city": "San Francisco", "zip_code": "94105"}),
            ("lookup_serviceability", {}),
        ),
        writer=say("Thanks, I've got your address."),
    ):
        out = s.run_turn("1 Market St, San Francisco, 94105")

    cart = s.snapshot()["cart"]
    assert cart["address"]["zip_code"] == "94105"
    assert cart["serviceable"] is True
    assert cart["step"] == "collecting_delivery"
    # a checkout block reflects the live cart
    assert any(b["kind"] == "checkout" for b in out["blocks"])


def test_empty_cart_hides_checkout_tool(override):
    """The empty-cart guard (tool ``prepare``) removes checkout from the router's tools."""
    seen: dict[str, list[str]] = {}

    def spy_orchestrator(messages, info):
        seen["tools"] = [t.name for t in info.function_tools]
        from pydantic_ai.messages import ModelResponse, TextPart

        return ModelResponse(parts=[TextPart("DONE")])

    s = ShoppingSession()  # empty cart
    with override(orchestrator=spy_orchestrator, writer=say("Hi.")):
        s.run_turn("hello")
    assert "checkout" not in seen["tools"]
    assert "product_rec" in seen["tools"]


def test_input_guardrail_blocks_before_any_model_call(override):
    from pydantic_agent_v1.runtime.guardrails import Blocklist

    s = ShoppingSession(input_rules=[Blocklist(phrases=["forbidden"], message="I can't help with that.")])
    # no override needed: a blocked turn never reaches a model
    out = s.run_turn("this is forbidden")
    assert out["message"] == "I can't help with that."
    guards = [e for e in out["events"] if e["type"] == "guardrail"]
    assert guards and guards[0]["rule"] == "blocklist"
