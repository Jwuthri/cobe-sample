"""Shared pytest config for the agent_agno_v1 suite.

These tests are hermetic — no real LLM call. The session test drives the streaming
pipeline with a scripted fake Agno event stream (see ``fakes.py``).
"""

from __future__ import annotations

import os

# Belt-and-braces: agent_agno_v1/__init__ also sets this, but keep CI offline even
# if a test imports agno before the package.
os.environ.setdefault("AGNO_TELEMETRY", "false")
