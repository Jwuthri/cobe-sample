"""The leaf registry — the data the graph + supervisor are generated from."""

from __future__ import annotations

from agent_v4 import ids
from agent_v4.leaves import (
    CHECKOUT_CONFIG,
    LEAF_NAMES,
    LEAVES,
    ORDER_STATUS_CONFIG,
    PRODUCT_REC_CONFIG,
    make_checkout_wrapper,
    make_order_status_wrapper,
    make_product_rec_wrapper,
    routing_catalog,
)


def _tool_names(cfg) -> set[str]:
    return {t.name for t in cfg.tools}


def test_leaf_names_match_ids():
    assert set(LEAF_NAMES) == {ids.CHECKOUT, ids.PRODUCT_REC, ids.ORDER_STATUS}
    # The registry is the single source of the topology.
    assert [s.name for s in LEAVES] == LEAF_NAMES


def test_checkout_config_has_all_constrained_tools_and_skills():
    # 13 constrained checkout tools, 5 ordered skills (matches v2 sops/checkout).
    assert len(CHECKOUT_CONFIG.tools) == 13
    assert len(CHECKOUT_CONFIG.skills) == 5
    skill_names = {s.name for s in CHECKOUT_CONFIG.skills}
    assert {"collect_identity", "collect_payment"} <= skill_names
    assert {"confirm_checkout", "set_address", "get_cart_summary"} <= _tool_names(CHECKOUT_CONFIG)
    assert [m.name for m in CHECKOUT_CONFIG.middleware] == ["log_tool_calls"]


def test_product_rec_config_tools():
    assert _tool_names(PRODUCT_REC_CONFIG) == {
        "search_products",
        "get_product",
        "check_serviceability",
        "add_item",
    }
    # product_rec is gateless — no skills.
    assert PRODUCT_REC_CONFIG.skills == []


def test_order_status_config_tools():
    assert _tool_names(ORDER_STATUS_CONFIG) == {"get_order_status", "list_recent_orders"}


def test_only_checkout_needs_a_checkpointer():
    by_name = {s.name: s for s in LEAVES}
    assert by_name[ids.CHECKOUT].needs_checkpointer is True
    assert by_name[ids.PRODUCT_REC].needs_checkpointer is False
    assert by_name[ids.ORDER_STATUS].needs_checkpointer is False
    # All leaves get the long-term memory store.
    assert all(s.needs_store for s in LEAVES)


def test_routing_catalog_describes_every_leaf():
    cat = routing_catalog()
    for name in LEAF_NAMES:
        assert name in cat


def test_wrapper_factories_return_callables():
    sentinel_agent = object()
    assert callable(make_checkout_wrapper(sentinel_agent))
    assert callable(make_product_rec_wrapper(sentinel_agent))
    assert callable(make_order_status_wrapper(sentinel_agent))
