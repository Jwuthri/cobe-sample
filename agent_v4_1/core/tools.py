"""Tool resolution: registry lookup + declarative HTTP-tool compilation.

The HTTP compiler is ported verbatim from agent_v4's (it's correct and tested):
``{placeholder}`` tokens in url/headers are filled from the model's args and
stripped from the request body so secrets passed in a header don't leak into the
payload.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
from langchain_core.tools import StructuredTool

from agent_v4_1.core.config import HttpToolSpec, RegistryToolSpec, ToolSpec
from agent_v4_1.core.registry import TOOLS

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


def compile_http_tool(spec: HttpToolSpec) -> StructuredTool:
    """Compile an :class:`HttpToolSpec` into a callable LangChain tool."""
    consumed = _template_keys(spec.url, *spec.headers.values())

    def _call(**kwargs: Any) -> str:
        url = _format_template(spec.url, kwargs)
        headers = {k: _format_template(v, kwargs) for k, v in spec.headers.items()}
        # Don't echo placeholder-only args (e.g. api_token) back in the payload.
        payload = {k: v for k, v in kwargs.items() if k not in consumed}
        with httpx.Client(timeout=spec.timeout_s) as client:
            if spec.method == "GET":
                response = client.get(url, headers=headers, params=payload)
            else:
                response = client.request(spec.method, url, headers=headers, json=payload)
        response.raise_for_status()
        return response.text

    return StructuredTool.from_function(
        func=_call,
        name=spec.name,
        description=spec.description or f"HTTP {spec.method} tool",
        args_schema=spec.parameters or None,
    )


def resolve_tools(specs: list[ToolSpec]) -> list[Any]:
    """Resolve a config's tool specs into concrete tool objects."""
    out: list[Any] = []
    for spec in specs:
        if isinstance(spec, RegistryToolSpec):
            out.append(TOOLS.get(spec.name))
        else:  # HttpToolSpec
            out.append(compile_http_tool(spec))
    return out


__all__ = ["compile_http_tool", "resolve_tools"]
