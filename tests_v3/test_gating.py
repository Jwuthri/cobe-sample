"""Skill-gating — re-implemented as an Agno tool hook (ports test_skills_gating.py).

In v2 each constrained tool called ``_require_skill``. In v3 the gating is
centralized in ``skill_gate_hook`` (attached at the agent level), which also
records skill loads when the agent calls ``get_skill_instructions``.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent_v3 import gating
from agent_v3.checkout import CartService
from agent_v3.tools.checkout_tools import add_item, confirm_checkout, set_customer


def _rc(skills_loaded=None, cart_service=None):
    return SimpleNamespace(
        dependencies={"skills_loaded": list(skills_loaded or []), "cart_service": cart_service, "store": None},
        session_state={"skills_loaded": list(skills_loaded or [])},
        user_id="u",
        session_id="s",
    )


def _passthrough(**kwargs):
    return "REAL_TOOL_RAN"


# ----- the hook -----
def test_gate_refuses_set_address_without_skill():
    rc = _rc(skills_loaded=[])
    out = gating.skill_gate_hook("set_address", _passthrough, {"street": "x"}, run_context=rc)
    assert out.startswith("Error: this tool requires the 'collect-address' skill")


def test_get_skill_instructions_records_load_and_unlocks():
    rc = _rc(skills_loaded=[])
    gating.skill_gate_hook(
        "get_skill_instructions", _passthrough, {"skill_name": "collect-address"}, run_context=rc
    )
    assert "collect-address" in rc.dependencies["skills_loaded"]
    out = gating.skill_gate_hook("set_address", _passthrough, {"street": "x"}, run_context=rc)
    assert out == "REAL_TOOL_RAN"


def test_ungated_tool_always_passes():
    rc = _rc(skills_loaded=[])
    assert gating.skill_gate_hook("add_item", _passthrough, {"product_id": "P-1"}, run_context=rc) == "REAL_TOOL_RAN"


# ----- the checkout tools themselves (gating now lives in the hook, not the body) -----
def test_set_customer_mutates_cart_via_run_context():
    svc = CartService()
    rc = _rc(cart_service=svc)
    out = set_customer("A", "B", run_context=rc)
    assert "Customer set to A B" in out
    assert svc.cart.customer.first_name == "A"


def test_add_item_via_run_context():
    svc = CartService()
    rc = _rc(cart_service=svc)
    out = add_item("P-2", run_context=rc)
    assert "Added" in out
    assert len(svc.cart.items) == 1


def test_confirm_refuses_when_not_ready():
    svc = CartService()
    rc = _rc(cart_service=svc)
    out = confirm_checkout(run_context=rc)
    assert "blockers" in out
    assert not svc.cart.confirmed
