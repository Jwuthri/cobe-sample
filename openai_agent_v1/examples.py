"""A reference agent config, in the exact shape a tenant writes.

This is the contract :mod:`openai_agent_v1.core.config` is designed around — the
acceptance test asserts ``AgentConfig.model_validate(EXAMPLE_AGENT_CONFIG)``
succeeds verbatim. It is a *living doc*, not a runnable agent: it references
registry tools (``check_order_status`` / ``create_support_ticket``) that the
shopping platform doesn't register, so it validates always but only *builds* once
those tools are stubbed (see ``tests_openai_v1/test_factory.py``). Don't "fix" the
missing tools — that's the point.
"""

from __future__ import annotations

EXAMPLE_AGENT_CONFIG = {
    "name": "Acme Support",
    "description": "Customer support agent with safety and session memory.",
    "system_prompt": "You are a friendly, professional customer support agent...",
    "instructions": ["Be concise and empathetic.", "response in english."],
    "model": {"provider_model": "openai:gpt-5-mini", "temperature": 0.0},
    "skills": [
        {
            "kind": "custom",
            "name": "checkout_flow",
            "description": "use when the user ask for checkout",
            "skill": "long text",
        }
    ],
    "tools": [
        {"kind": "registry", "name": "check_order_status"},
        {"kind": "registry", "name": "create_support_ticket"},
        {
            "kind": "http",
            "name": "create_zendesk_ticket",
            "description": "Open a support ticket in the customer's Zendesk.",
            "method": "POST",
            "url": "https://acme.zendesk.com/api/v2/tickets",
            "headers": {"Authorization": "Bearer {api_token}"},
            "parameters": {
                "type": "object",
                "properties": {
                    "api_token": {"type": "string"},
                    "subject": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "high"]},
                },
                "required": ["subject"],
            },
        },
    ],
    "guardrails": [
        {"type": "pii", "action": "redact", "params": {"entity": "email"}},
        {
            "type": "blocklist",
            "action": "block",
            "on_input": True,
            "message": (
                "I'm only able to help with product and billing support. "
                "I can't provide legal or medical advice."
            ),
            "params": {"phrases": ["sue", "lawsuit", "medical diagnosis"]},
        },
        {
            "type": "llm_judge",
            "action": "block",
            "on_input": True,
            "message": "Sorry, I can't discuss that topic.",
            "params": {
                "policy": "Do not answer anything about Elon Musk.",
                "model": "openai:gpt-5-nano",
            },
        },
    ],
    "middleware": [
        {"name": "model_call_counter", "params": {}},
        {"name": "max_turns", "params": {"max_turns": 30}},
        {"name": "log_tool_calls", "params": {"log_prefix": "acme-support"}},
    ],
    "output_format": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Brief summary."},
            "status": {
                "type": "string",
                "enum": ["open", "pending_customer", "resolved", "escalated"],
            },
            "resolution": {
                "type": "object",
                "properties": {
                    "next_steps": {"type": "array", "items": {"type": "str"}},
                },
            },
        },
        "required": ["summary", "status"],
    },
}

__all__ = ["EXAMPLE_AGENT_CONFIG"]
