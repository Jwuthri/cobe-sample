"""Skills middleware — the available-skills block + tool gating via ``unlocks``."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from lg_agent.core.skills import Skill, SkillsMiddleware, render_available_block
from lg_agent.shopping.agents.subagents.checkout_skills import CHECKOUT_SKILLS, STEP_SKILL


class _Req:
    """Minimal stand-in for a model-call request (state / messages / tools / override)."""

    def __init__(self, tools, skills_loaded):
        self.tools = tools
        self.state = {"skills_loaded": skills_loaded}
        self.messages: list = []
        self.applied: dict = {}

    def override(self, **kw):
        self.applied = kw
        if "tools" in kw:
            self.tools = kw["tools"]
        if "messages" in kw:
            self.messages = kw["messages"]
        return self


def _tool(name):
    return type("T", (), {"name": name})()


def test_available_block_marks_loaded():
    skills = [Skill(name="a", description="da", content="x"), Skill(name="b", description="db", content="y")]
    block = render_available_block(skills, loaded=["a"])
    assert "a (loaded): da" in block
    assert "b: db" in block  # not loaded → no marker


def test_gated_tool_hidden_until_skill_loaded():
    mw = SkillsMiddleware([Skill(name="collect_identity", content="x", unlocks=["set_customer"])])
    setc, summary = _tool("set_customer"), _tool("get_cart_summary")

    # not loaded → the gated tool is removed, the ungated one stays
    req = _Req([setc, summary], skills_loaded=[])
    mw._apply(req)
    assert [t.name for t in req.tools] == ["get_cart_summary"]
    # the available-skills block is prepended as a system message
    assert isinstance(req.messages[0], SystemMessage)

    # loaded → the gated tool is now visible
    req2 = _Req([setc, summary], skills_loaded=["collect_identity"])
    mw._apply(req2)
    assert sorted(t.name for t in req2.tools) == ["get_cart_summary", "set_customer"]


def test_no_unlocks_means_no_gating():
    # pure instruction-bundle skills never touch the tool list
    mw = SkillsMiddleware([Skill(name="info", content="x")])  # no unlocks
    tools = [_tool("set_customer"), _tool("anything")]
    req = _Req(tools, skills_loaded=[])
    mw._apply(req)
    assert sorted(t.name for t in req.tools) == ["anything", "set_customer"]
    assert "tools" not in req.applied  # tool list untouched


def test_checkout_skills_cover_every_step_and_tool():
    # every step the user passes through maps to a real skill...
    skill_names = {s["name"] for s in CHECKOUT_SKILLS}
    assert set(STEP_SKILL.values()) <= skill_names
    # ...and every gated checkout tool is unlocked by exactly one skill
    unlocked = [t for s in CHECKOUT_SKILLS for t in s["unlocks"]]
    assert len(unlocked) == len(set(unlocked))  # no tool gated by two skills
    assert {"set_customer", "set_address", "confirm_checkout"} <= set(unlocked)
