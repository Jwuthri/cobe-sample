"""Tool resolution: registry lookup + declarative HTTP-tool compilation.

The HTTP compiler ports agent_v4_1's: ``{placeholder}`` tokens in url/headers are
filled from the model's args and stripped from the request body so secrets passed
in a header don't leak into the payload. It is built on the SDK's ``FunctionTool``
(constructed directly from the spec's JSON-Schema ``parameters``, since the args
schema is declared, not inferred from a Python signature).
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
from agents import FunctionTool, RunContextWrapper

from openai_agent_v1.core.config import HttpToolSpec, RegistryToolSpec, ToolSpec
from openai_agent_v1.core.registry import TOOLS

_PLACEHOLDER_KEYS = re.compile(r"\{(\w+)\}")


def _template_keys(*templates: str) -> set[str]:
    keys: set[str] = set()
    for t in templates:
        keys.update(_PLACEHOLDER_KEYS.findall(t))
    return keys


def _format_template(template: str, args: dict[str, Any]) -> str:
    try:
        return template.format(**args)
    except (KeyError, IndexError):
        return template


def compile_http_tool(spec: HttpToolSpec) -> FunctionTool:
    """Compile an :class:`HttpToolSpec` into a callable SDK ``FunctionTool``."""
    consumed = _template_keys(spec.url, *spec.headers.values())

    async def _on_invoke(_ctx: RunContextWrapper[Any], args_json: str) -> str:
        kwargs = json.loads(args_json) if args_json else {}
        url = _format_template(spec.url, kwargs)
        headers = {k: _format_template(v, kwargs) for k, v in spec.headers.items()}
        # Don't echo placeholder-only args (e.g. api_token) back in the payload.
        payload = {k: v for k, v in kwargs.items() if k not in consumed}
        async with httpx.AsyncClient(timeout=spec.timeout_s) as client:
            if spec.method == "GET":
                response = await client.get(url, headers=headers, params=payload)
            else:
                response = await client.request(spec.method, url, headers=headers, json=payload)
        response.raise_for_status()
        return response.text

    return FunctionTool(
        name=spec.name,
        description=spec.description or f"HTTP {spec.method} tool",
        params_json_schema=spec.parameters or {"type": "object", "properties": {}},
        on_invoke_tool=_on_invoke,
        strict_json_schema=False,
    )


def resolve_tools(specs: list[ToolSpec]) -> list[Any]:
    """Resolve a config's tool specs into concrete SDK tool objects."""
    out: list[Any] = []
    for spec in specs:
        if isinstance(spec, RegistryToolSpec):
            out.append(TOOLS.get(spec.name))
        else:  # HttpToolSpec
            out.append(compile_http_tool(spec))
    return out


__all__ = ["compile_http_tool", "resolve_tools"]
