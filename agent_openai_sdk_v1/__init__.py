"""agent_openai_sdk_v1 — the agent_v4_1 shopping assistant, rebuilt from scratch on
the OpenAI Agents Python SDK.

A multi-agent shopping assistant with a careful checkout flow and true token
streaming. The architecture is three clean layers:

  * :mod:`agent_openai_sdk_v1.domain`  — the store, as pure logic (the behavioral spec);
  * :mod:`agent_openai_sdk_v1.runtime` — thin generic glue (context, events, delegation);
  * :mod:`agent_openai_sdk_v1.agents`  — five agents, one self-contained file each.

The :class:`ShoppingSession` ties them together into a streaming turn. See the
package README for the full map.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load the repo-root .env (OPENAI_API_KEY + model overrides) before anything resolves
# a model name. Importing any submodule triggers this package first, so it always wins.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))

from agent_openai_sdk_v1.session import ShoppingSession  # noqa: E402

__all__ = ["ShoppingSession"]
