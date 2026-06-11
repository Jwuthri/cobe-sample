"""agent_agno_v1 — agent_v4_1's shopping assistant, rebuilt on Agno (2.6.x).

A coordinate-mode Agno ``Team`` (the "speaking supervisor") routes the user's
message to three member sub-agents (product_rec / checkout / order_status) over a
single live cart, then authors the user-facing reply — streamed token-by-token.
The cart invariant gates checkout, and structured cards are built deterministically
(the hallucination firewall), exactly as in agent_v4_1.

Importing this package disables Agno's anonymous telemetry and loads the repo
``.env`` (so ``OPENAI_API_KEY`` / ``AGENT_*_MODEL`` are available).
"""

from __future__ import annotations

import os

# Disable Agno's telemetry phone-home BEFORE any agno import (env wins over the
# constructor flag in agno 2.6.x). setdefault so an explicit override still works.
os.environ.setdefault("AGNO_TELEMETRY", "false")

try:  # auto-load the repo .env, like agent_v2/agent_v4_1 do
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # pragma: no cover - dotenv is optional
    pass

__all__ = ["__version__"]
__version__ = "0.1.0"
