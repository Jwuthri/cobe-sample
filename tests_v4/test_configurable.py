"""The declarative agent builder — registries, ModelConfig, tool/skill
compilation, HttpTool, and build_agent.

These are the tests for the *new* layer agent_v4 adds on top of v2. No
network: ChatOpenAI construction and create_agent compilation are local;
HttpTool execution is exercised against a fake httpx client.
"""

from __future__ import annotations

import pytest
from agent_v4 import configurable as cfgmod
from agent_v4.configurable import (
    AgentConfig,
    HttpTool,
    MiddlewareSpec,
    ModelConfig,
    Registry,
    RegistryTool,
    SkillSpec,
    _compose_prompt,
    _template_keys,
    build_agent,
    build_catalog,
)
from agent_v4.registry_defaults import register_platform_defaults


# ----- registry -----
def test_registry_get_has_and_unknown():
    r = Registry("widget")
    sentinel = object()
    r.register("a", sentinel, label="A")
    assert r.has("a")
    assert r.get("a") is sentinel
    assert not r.has("b")
    with pytest.raises(ValueError):
        r.get("b")
    assert {"name": "a", "label": "A"} in r.catalog()


def test_platform_defaults_register_tools_and_skills():
    register_platform_defaults()
    assert cfgmod.TOOL_REGISTRY.has("add_item")
    assert cfgmod.TOOL_REGISTRY.has("search_products")
    assert cfgmod.SKILL_REGISTRY.has("collect_identity")
    assert cfgmod.MIDDLEWARE_REGISTRY.has("log_tool_calls")


# ----- model config (fixes the doc bug: temperature/max_tokens were ignored) -----
def test_model_config_applies_temperature_and_max_tokens():
    m = ModelConfig(model="gpt-4.1-mini", temperature=0.7, max_tokens=128).build()
    assert m.temperature == 0.7
    assert m.max_tokens == 128


def test_model_config_strips_openai_prefix():
    m = ModelConfig(model="openai:gpt-4.1-mini").build()
    assert m.model_name == "gpt-4.1-mini"


def test_model_config_defaults_to_env_model():
    from agent_v4.llm import model_name

    m = ModelConfig().build()
    assert m.model_name == model_name()


# ----- prompt composition -----
def test_compose_prompt_appends_instructions():
    cfg = AgentConfig(
        name="t",
        system_prompt="Base prompt.",
        instructions=["Be concise.", "No emoji."],
    )
    prompt = _compose_prompt(cfg)
    assert prompt.startswith("Base prompt.")
    assert "## Additional instructions" in prompt
    assert "- Be concise." in prompt
    assert "- No emoji." in prompt


def test_compose_prompt_without_instructions_is_just_the_prompt():
    cfg = AgentConfig(name="t", system_prompt="  Hello.  ")
    assert _compose_prompt(cfg) == "Hello."


# ----- build_agent compiles a runnable from config -----
def test_build_agent_compiles_runnable():
    register_platform_defaults()
    cfg = AgentConfig(
        name="mini",
        system_prompt="You are a tiny test agent.",
        tools=[RegistryTool(name="search_products")],
        middleware=[MiddlewareSpec(name="log_tool_calls")],
    )
    agent = build_agent(cfg)
    assert hasattr(agent, "invoke")
    assert hasattr(agent, "stream")


def test_build_agent_attaches_skills_middleware_with_load_skill_tool():
    register_platform_defaults()
    cfg = AgentConfig(
        name="skilled",
        system_prompt="x",
        skills=[SkillSpec(name="collect_identity")],
    )
    mw = cfgmod._compile_skills_middleware(cfg)
    assert mw is not None
    # The SkillsMiddleware exposes a load_skill tool.
    tool_names = {t.name for t in mw.tools}
    assert "load_skill" in tool_names


def test_build_agent_no_skills_means_no_skills_middleware():
    cfg = AgentConfig(name="plain", system_prompt="x")
    assert cfgmod._compile_skills_middleware(cfg) is None


# ----- catalog -----
def test_build_catalog_shape():
    register_platform_defaults()
    cat = build_catalog()
    tool_names = {t["name"] for t in cat.tools}
    assert "add_item" in tool_names and "get_order_status" in tool_names
    assert {"pii", "blocklist", "llm_judge"} <= {g["name"] for g in cat.guardrails}
    assert {"collect_identity"} <= {s["name"] for s in cat.skills}


# ----- HttpTool: declarative compile + the payload-leak fix -----
def test_template_keys_extracts_placeholders():
    keys = _template_keys("https://api/{base}/x", "Bearer {api_token}")
    assert keys == {"base", "api_token"}


class _FakeResp:
    text = '{"ok": true}'

    def raise_for_status(self):
        return None


class _FakeClient:
    last: dict = {}

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def request(self, method, url, headers=None, json=None):
        _FakeClient.last = {"method": method, "url": url, "headers": headers, "json": json}
        return _FakeResp()

    def get(self, url, headers=None, params=None):
        _FakeClient.last = {"method": "GET", "url": url, "headers": headers, "params": params}
        return _FakeResp()


def test_http_tool_fills_placeholders_and_strips_them_from_payload(monkeypatch):
    monkeypatch.setattr(cfgmod.httpx, "Client", _FakeClient)
    spec = HttpTool(
        name="create_ticket",
        description="open a ticket",
        method="POST",
        url="https://acme.example/{base}/tickets",
        headers={"Authorization": "Bearer {api_token}"},
        parameters={
            "type": "object",
            "properties": {
                "base": {"type": "string"},
                "api_token": {"type": "string"},
                "subject": {"type": "string"},
            },
        },
    )
    tool = cfgmod._compile_http_tool(spec)
    out = tool.invoke({"base": "v2", "api_token": "secret123", "subject": "help"})
    assert out == '{"ok": true}'
    sent = _FakeClient.last
    # URL + header placeholders were filled...
    assert sent["url"] == "https://acme.example/v2/tickets"
    assert sent["headers"]["Authorization"] == "Bearer secret123"
    # ...and the placeholder-consumed args are NOT echoed in the body.
    assert "api_token" not in sent["json"]
    assert "base" not in sent["json"]
    assert sent["json"] == {"subject": "help"}
