"""lg_agent_v2 — the pydantic_agent_v1 shopping assistant, rebuilt on LangChain +
LangGraph.

A multi-agent shopping assistant with a careful checkout flow and true token
streaming. It is a clean-room port of ``pydantic_agent_v1``: the SAME three-layer
architecture and the SAME frontend wire contract, with only the harness swapped from
Pydantic AI to LangChain ``create_agent`` graphs:

  * :mod:`lg_agent_v2.domain`  — the store, as pure logic (the behavioral spec);
  * :mod:`lg_agent_v2.runtime` — thin generic glue (deps/context, events, delegation,
    the middleware primitives that stand in for Pydantic AI natives);
  * :mod:`lg_agent_v2.agents`  — five agents, one self-contained file each.

The :class:`ShoppingSession` ties them together into a streaming turn. See the package
README for the full map.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load the repo-root .env (OPENAI_API_KEY + model overrides) before anything resolves
# a model name. Importing any submodule triggers this package first, so it always wins.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))

from lg_agent_v2.session import ShoppingSession  # noqa: E402

__all__ = ["ShoppingSession"]
