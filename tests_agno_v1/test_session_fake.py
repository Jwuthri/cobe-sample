"""The streaming session pipeline, exercised end-to-end with a fake Agno stream.

No real LLM: a :class:`FakeTeam` yields the exact agno event taxonomy and mutates
the shared cart as its tools 'run'. This verifies the event bridge, StepResult
distillation, deterministic blocks, the leader-vs-member token discrimination,
the guardrail pre-flight, and the snapshot contract.
"""

from __future__ import annotations

from agent_agno_v1.core.config import GuardrailSpec
from agent_agno_v1.core.guardrails import compile_input_rules
from agent_agno_v1.shopping.domain import CartService
from agent_agno_v1.shopping.session import ShoppingSession
from tests_agno_v1.fakes import Delegation, FakeTeam, ToolCall

HOODIE_LINE = "P-2: Black Hoodie — $49.99 [apparel, hoodie, black]"


def _session(cart_service: CartService, team, **kw) -> ShoppingSession:
    return ShoppingSession(cart_service=cart_service, team=team, session_id="t", **kw)


def _types(events) -> list[str]:
    return [e["type"] for e in events]


# =============================================================================
# browse (product_rec, no cart change)
# =============================================================================
def test_browse_streams_tokens_and_builds_product_block():
    cs = CartService()
    team = FakeTeam(
        delegations=[
            Delegation(
                member_id="product-rec",
                tools=[ToolCall("search_products", {"query": "hoodies"}, result=HOODIE_LINE)],
                member_reply="I found the Black Hoodie P-2.",  # member chatter, NOT user-facing
            )
        ],
        leader_reply="Here are the hoodies I found.",
    )
    out = _session(cs, team).run_turn("show me hoodies")
    events = out["events"]
    types = _types(events)

    # routing + tool + step + token + bot all present, in pipeline order
    assert "router" in types and "tool_start" in types and "tool_end" in types
    assert "step" in types and "token" in types and "bot" in types
    # the router target was canonicalised back to the sop vocabulary
    router = next(e for e in events if e["type"] == "router")
    assert router["target"] == "product_rec"
    # the LEADER reply streamed as tokens; the MEMBER chatter did NOT
    streamed = "".join(out["tokens"])
    assert "hoodies I found" in streamed
    assert "P-2" not in streamed and "found the Black" not in streamed
    assert out["message"] == "Here are the hoodies I found."
    # deterministic product block with the verbatim id/price
    blocks = out["blocks"]
    assert [b["kind"] for b in blocks] == ["product_reco"]
    assert blocks[0]["products"][0]["id"] == "P-2"
    assert blocks[0]["products"][0]["price"] == "49.99"


def test_add_item_mutates_cart_and_signals_checkout():
    cs = CartService()
    team = FakeTeam(
        delegations=[
            Delegation(
                member_id="product-rec",
                tools=[
                    ToolCall(
                        "add_item",
                        {"product_id": "P-2"},
                        result="Added 1 × Black Hoodie.",
                        mutate=lambda: cs.add_item("P-2"),
                    )
                ],
            )
        ],
        leader_reply="Added the Black Hoodie to your cart.",
    )
    out = _session(cs, team).run_turn("add the black hoodie")
    # cart actually changed
    assert [i.product_id for i in cs.cart.items] == ["P-2"]
    # step says added + points to checkout next
    step = next(e for e in out["events"] if e["type"] == "step")
    assert "added" in step["summary"].lower()
    assert step["next_sop"] == "checkout"
    # product block records the added id
    assert out["blocks"][0]["added_ids"] == ["P-2"]


# =============================================================================
# checkout (cart mutation + asks)
# =============================================================================
def test_checkout_block_lists_next_asks():
    cs = CartService()
    cs.add_item("P-2")  # item already in cart
    team = FakeTeam(
        delegations=[
            Delegation(
                member_id="checkout",
                tools=[
                    ToolCall(
                        "set_customer",
                        {"first_name": "Ada", "last_name": "Lovelace"},
                        result="Customer set to Ada Lovelace.",
                        mutate=lambda: cs.set_customer("Ada", "Lovelace"),
                    )
                ],
            )
        ],
        leader_reply="Thanks Ada. What is your shipping address?",
    )
    out = _session(cs, team).run_turn("my name is Ada Lovelace")
    assert cs.cart.customer.first_name == "Ada"
    block = out["blocks"][0]
    assert block["kind"] == "checkout"
    # next step is address → the checkout block surfaces those asks
    assert any("street" in a or "city" in a or "zip" in a for a in block["asks"])
    assert block["confirmed"] is False


def test_confirm_gate_blocks_without_full_cart():
    """Even if the model 'confirms', the cart invariant refuses until ready."""
    cs = CartService()
    cs.add_item("P-2")
    # confirm tool runs against the real cart_service → returns an error string
    from agent_agno_v1.shopping.tools import confirm_checkout  # noqa: F401  (registered)

    team = FakeTeam(
        delegations=[
            Delegation(
                member_id="checkout",
                tools=[ToolCall("confirm_checkout", {}, result="error: cannot confirm — blockers: missing_identity")],
            )
        ],
        leader_reply="I still need a few details before placing the order.",
    )
    out = _session(cs, team).run_turn("place the order")
    assert cs.cart.confirmed is False
    assert out["blocks"][0]["confirmed"] is False


# =============================================================================
# guardrail pre-flight
# =============================================================================
def test_input_guardrail_blocks_before_team_runs():
    cs = CartService()

    class ExplodingTeam:
        members: list = []

        def arun(self, *a, **k):
            raise AssertionError("team must not run when input is blocked")

    rules = compile_input_rules(
        [GuardrailSpec(type="blocklist", action="block", message="No.", params={"phrases": ["forbidden"]})]
    )
    out = _session(cs, ExplodingTeam(), input_rules=rules).run_turn("this is forbidden")
    types = _types(out["events"])
    assert "guardrail" in types
    assert out["message"] == "No."
    assert "router" not in types and "token" not in types


# =============================================================================
# smalltalk (no delegation, no blocks)
# =============================================================================
def test_smalltalk_streams_reply_with_no_blocks():
    cs = CartService()
    team = FakeTeam(delegations=[], leader_reply="Hi! I can help you find products and place orders.")
    out = _session(cs, team).run_turn("hello")
    assert out["blocks"] == []
    assert "step" not in _types(out["events"])
    assert "Hi!" in out["message"]
    assert out["tokens"]  # streamed
