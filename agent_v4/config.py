"""Shared runtime flags (loaded from env / ``.env``)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Same load order as ``agent_v4.__init__`` so ``server.main`` picks up .env.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def debug_enabled() -> bool:
    return os.getenv("AGENT_V4_DEBUG", "false").strip().lower() in _TRUTHY


def setup_debug_logging() -> None:
    """When debug is on: Rich event lines on stderr + quiet third-party loggers."""
    if not debug_enabled():
        return
    from agent_v4.debug_log import quiet_noisy_loggers

    quiet_noisy_loggers()
