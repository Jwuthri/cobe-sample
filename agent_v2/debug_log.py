"""Rich stderr debug lines when ``AGENT_V2_DEBUG=true``.

Matches the CLI / web event-stream color scheme so tool/skill/SSE traces
are readable in the uvicorn terminal.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.text import Text

from agent_v2.config import debug_enabled

# Keep in sync with ``agent_v2/cli.py`` EventLog styles.
_KIND_STYLES: dict[str, str] = {
    "USER": "cyan bold",
    "ROUTER": "blue",
    "AGENT": "magenta",
    "SKILL": "yellow",
    "TOOL": "green",
    "RESULT": "green dim",
    "STEP": "orange3",
    "WRITER": "cyan dim",
    "GATE": "red",
    "VALIDATOR": "yellow dim",
    "BOT": "cyan",
    "STATE": "dim",
    "ERROR": "red bold",
}

_console: Console | None = None
_MAX_RESULT = 320


def _get_console() -> Console:
    global _console
    if _console is None:
        _console = Console(stderr=True, highlight=False, soft_wrap=True)
    return _console


def emit(kind: str, body: str) -> None:
    if not debug_enabled():
        return
    ts = datetime.now().strftime("%H:%M:%S")
    style = _KIND_STYLES.get(kind, "white")
    line = Text.assemble(
        (f"{ts} ", "dim"),
        (f"{kind:9s}", style),
        " ",
        (body, ""),
    )
    _get_console().print(line)


def _short(text: str, limit: int = _MAX_RESULT) -> str:
    one_line = text.replace("\n", " ⏎ ")
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit] + "…"


def _format_args(args: dict[str, Any]) -> str:
    parts: list[str] = []
    for k, v in args.items():
        if k == "tool_call_id":
            continue
        if isinstance(v, str) and len(v) > 80:
            v = v[:77] + "…"
        parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def tool_start(name: str, args: dict[str, Any] | None = None) -> None:
    args = args or {}
    if name == "load_skill":
        emit("SKILL", f"load → {args.get('skill_name', '?')}")
        return
    emit("TOOL", f"{name}({_format_args(args)})")


def tool_end(name: str, result: str = "") -> None:
    if not result:
        return
    emit("RESULT", f"↳ {name}: {_short(result)}")


def graph(node: str, detail: str) -> None:
    emit("AGENT", f"{node} — {detail}")


def sse(ev: dict[str, Any]) -> None:
    """Pretty-print one SSE / UI event dict (same shapes as ``server/main``)."""
    if not debug_enabled():
        return
    t = ev.get("type")
    if t == "user":
        emit("USER", str(ev.get("content", "")))
    elif t == "router":
        iter_n = ev.get("iteration", 0)
        suffix = f" (iter {iter_n})" if iter_n else ""
        emit("ROUTER", f"→ {ev.get('target', '?')}{suffix}")
    elif t == "agent":
        emit("AGENT", f"{ev.get('node', '?')} finished")
    elif t in ("skill", "tool_start", "tool_end"):
        # Already printed by ``log_tool_calls`` — skip duplicate SSE lines.
        return
    elif t == "step":
        parts = [f"{ev.get('sop', '?')}: {ev.get('summary', '')}"]
        asks = ev.get("asks") or []
        if asks:
            parts.append(f"asks=[{', '.join(asks)}]")
        if ev.get("next_sop"):
            parts.append(f"→ {ev['next_sop']}")
        emit("STEP", " ".join(parts))
    elif t == "writer":
        emit("WRITER", _short(str(ev.get("draft", "")), 200))
    elif t == "gate" and ev.get("rejected"):
        errs = ev.get("errors") or []
        emit("GATE", f"rejected: {'; '.join(errs)}")
    elif t == "validator":
        errs = ev.get("errors") or []
        if errs:
            emit("VALIDATOR", ", ".join(errs))
    elif t == "bot":
        emit("BOT", _short(str(ev.get("content", "")), 240))
    elif t == "state":
        snap = ev.get("snapshot") or {}
        cart = snap.get("cart") or {}
        emit(
            "STATE",
            f"step={cart.get('step', '?')} items={len(cart.get('items') or [])} "
            f"skills={snap.get('skills_loaded') or []}",
        )
    elif t == "error":
        emit("ERROR", str(ev.get("content", "")))
    elif t == "end":
        emit("STATE", "— end —")
    else:
        emit("STATE", _short(json.dumps(ev, default=str), 200))


def quiet_noisy_loggers() -> None:
    """Push library chatter to WARNING so only Rich lines show on stderr."""
    logging.basicConfig(level=logging.WARNING, force=True)
    for name in (
        "httpx",
        "httpcore",
        "openai",
        "openai._base_client",
        "langchain",
        "langchain_core",
        "langgraph",
        "uvicorn.access",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
