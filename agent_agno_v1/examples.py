"""A reference ``AgentConfig`` exercising every corner of the config schema.

``EXAMPLE_AGENT_CONFIG`` validates verbatim (see ``tests_agno_v1/test_config.py``)
and documents the declarative surface: a model spec, registry + declarative HTTP
tools, input/output guardrails, a tool-call cap, and a structured output schema.
The shopping demo itself uses a thinner slice of this (registry tools, no
guardrails — the cart invariant gates checkout).
"""

from __future__ import annotations

from typing import Any

EXAMPLE_AGENT_CONFIG: dict[str, Any] = {
    "name": "Concierge",
    "id": "concierge",
    "role": "Answer account questions and look things up over HTTP.",
    "system_prompt": "You are a helpful concierge. Use your tools to answer precisely.",
    "instructions": [
        "Prefer a tool call over guessing.",
        "Never reveal internal ids unless asked.",
    ],
    "model": {"provider_model": "openai:gpt-4.1-mini", "temperature": 0.2, "max_tokens": 800},
    "tools": [
        {"kind": "registry", "name": "search_products"},
        {
            "kind": "http",
            "name": "lookup_account",
            "description": "Look up an account by id via the billing API.",
            "method": "GET",
            "url": "https://api.example.com/accounts/{account_id}",
            "headers": {"Authorization": "Bearer {api_token}"},
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "description": "The account id."},
                    "api_token": {"type": "string", "description": "Bearer token (kept out of the body)."},
                },
                "required": ["account_id", "api_token"],
            },
        },
    ],
    "guardrails": [
        {
            "type": "blocklist",
            "action": "block",
            "on_input": True,
            "message": "I can't help with that.",
            "params": {"phrases": ["wire me your password"], "patterns": [r"\bssn\b"]},
        },
        {
            "type": "pii",
            "action": "redact",
            "on_input": True,
            "params": {"entities": ["email", "phone"]},
        },
    ],
    "tool_call_limit": 8,
    "output_format": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["answer"],
    },
}

__all__ = ["EXAMPLE_AGENT_CONFIG"]
