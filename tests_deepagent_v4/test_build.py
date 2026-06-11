"""Offline structural tests — assemble the specs/agent without any model call."""

from __future__ import annotations

import os
import pathlib

import pytest
from deepagents.middleware.skills import _parse_skill_metadata

from agent_deepagent_v4.agents import (
    build_checkout_subagent,
    build_order_status_subagent,
    build_product_rec_subagent,
    build_writer_subagent,
)
from agent_deepagent_v4.agents.orchestrator.agent import SKILLS_ROOT, build_orchestrator
from agent_deepagent_v4.config import load_env

load_env()


def _meta_field(meta, key):
    return meta[key] if isinstance(meta, dict) else getattr(meta, key)


def test_subagent_specs_have_required_fields():
    specs = [
        build_product_rec_subagent(),
        build_checkout_subagent(),
        build_order_status_subagent(),
        build_writer_subagent(),
    ]
    names = {s["name"] for s in specs}
    assert names == {"product-agent", "checkout-agent", "order-status-agent", "writer-agent"}
    for s in specs:
        assert s["name"] and s["description"] and s["system_prompt"]
        assert s["tools"], f"{s['name']} must declare scoped tools"
        assert ":" in s["model"], "model must be a provider:model string"


def test_checkout_has_no_add_item_tool():
    # Structural single-responsibility guard: checkout cannot add products.
    spec = build_checkout_subagent()
    tool_names = {t.name for t in spec["tools"]}
    assert "add_item" not in tool_names
    assert "confirm_checkout" in tool_names
    assert spec["skills"] == ["checkout"]


def test_product_agent_owns_cart_edits():
    spec = build_product_rec_subagent()
    tool_names = {t.name for t in spec["tools"]}
    assert {"add_item", "remove_item", "set_quantity", "search_products"} <= tool_names


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="constructing the chat models needs OPENAI_API_KEY",
)
def test_orchestrator_builds():
    agent = build_orchestrator()
    assert agent is not None
    # The task tool + workers should be discoverable; just confirm it compiled.
    assert hasattr(agent, "invoke")


def test_all_skill_files_valid():
    root = pathlib.Path(SKILLS_ROOT)
    skill_files = sorted(root.rglob("SKILL.md"))
    assert len(skill_files) >= 3
    for md in skill_files:
        meta = _parse_skill_metadata(md.read_text(), str(md), md.parent.name)
        assert meta is not None, f"{md} failed to parse"
        assert _meta_field(meta, "name") == md.parent.name
        assert _meta_field(meta, "description").strip()
