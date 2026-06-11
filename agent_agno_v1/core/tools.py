"""Tool resolution: registry lookup + declarative HTTP-tool compilation.

A config's ``tools`` list resolves to concrete Agno tool objects. ``registry``
specs are looked up by name; ``http`` specs are compiled into an Agno ``Function``
with an explicit JSON-Schema ``parameters`` (so the model can call a tenant API
declared purely as data). ``{placeholder}`` tokens in url/headers are filled from
the model's args and stripped from the request body, so a secret passed in a
header never leaks into the payload (ported from agent_v4_1, retargeted to Agno).
"""

from __future__ import annotations

import re
from typing import Any

import httpx
from agno.tools import Function

from agent_agno_v1.core.config import HttpToolSpec, RegistryToolSpec, ToolSpec
from agent_agno_v1.core.registry import TOOLS

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


def compile_http_tool(spec: HttpToolSpec) -> Function:
    """Compile an :class:`HttpToolSpec` into a callable Agno ``Function``."""
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

    return Function(
        name=spec.name,
        description=spec.description or f"HTTP {spec.method} tool",
        parameters=spec.parameters or {"type": "object", "properties": {}},
        entrypoint=_call,
        skip_entrypoint_processing=True,
    )


def resolve_tools(specs: list[ToolSpec]) -> list[Any]:
    """Resolve a config's tool specs into concrete Agno tool objects."""
    out: list[Any] = []
    for spec in specs:
        if isinstance(spec, RegistryToolSpec):
            out.append(TOOLS.get(spec.name))
        else:  # HttpToolSpec
            out.append(compile_http_tool(spec))
    return out


__all__ = ["compile_http_tool", "resolve_tools"]
