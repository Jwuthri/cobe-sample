"""Message helpers shared across the package."""

from __future__ import annotations

from typing import Any


def text_of(message: Any) -> str:
    """Extract plain text from a LangChain message.

    Modern models (e.g. gpt-5.4) return ``content`` as a list of typed blocks
    ``[{"type": "text", "text": "..."}]`` rather than a bare string; this
    flattens either shape to the human-readable text.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)
