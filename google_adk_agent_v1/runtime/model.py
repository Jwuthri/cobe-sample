"""Model resolution — one place that decides which LLM the agents talk to.

Google ADK speaks to Gemini natively (pass a model id string like
``"gemini-2.5-flash"``) and to everyone else through LiteLLM (wrap the id in
:class:`~google.adk.models.lite_llm.LiteLlm`). The name is resolved from the env so
the whole repo can be pointed at a different model without touching code:

    GOOGLE_ADK_AGENT_V1_MODEL  →  AGENT_V2_OPENAI_MODEL  →  gpt-5.4-mini

A bare id (no ``provider/`` prefix and not a ``gemini-*`` name) is assumed to be an
OpenAI model and is wrapped as ``LiteLlm("openai/<id>")`` so it runs against the same
``OPENAI_API_KEY`` the rest of the repo already uses. This keeps the ADK port a true
behavioral twin of ``pydantic_agent_v1`` (same model, same key) while remaining a
genuine Google-ADK build: point ``GOOGLE_ADK_AGENT_V1_MODEL`` at ``gemini-2.5-flash``
(with a ``GOOGLE_API_KEY`` set) and it runs natively on Gemini instead.
"""

from __future__ import annotations

import os

from google.adk.models import BaseLlm
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

_DEFAULT_MODEL = "gpt-5.4-mini"


def model_name() -> str:
    """The configured model id (provider-prefixed or bare)."""
    return (
        os.environ.get("GOOGLE_ADK_AGENT_V1_MODEL")
        or os.environ.get("AGENT_V2_OPENAI_MODEL")
        or _DEFAULT_MODEL
    )


def make_model() -> str | BaseLlm:
    """Resolve the env model id to something ``LlmAgent(model=...)`` accepts.

    * ``gemini-*`` → the bare string (ADK's native Gemini path);
    * ``provider/model`` → ``LiteLlm`` as-is (e.g. ``anthropic/claude-...``);
    * a bare id → ``LiteLlm("openai/<id>")`` (assumed OpenAI; runs on OPENAI_API_KEY).
    """
    name = model_name()
    if name.startswith("gemini-") or name.startswith("gemini/"):
        return name
    if "/" in name:
        return LiteLlm(model=name)
    return LiteLlm(model=f"openai/{name}")


MODEL_NAME = model_name()


def gen_config(temperature: float) -> types.GenerateContentConfig:
    """Build the per-agent generation config (ADK's analogue of ModelSettings).

    Only ``temperature`` varies between our agents (0.0 for the deterministic router
    + workers, a touch of warmth for the writer). The router additionally relies on
    its prompt + the per-turn delegation lock to route ONE worker per request, which
    is what Pydantic AI's ``parallel_tool_calls=False`` bought there.
    """
    return types.GenerateContentConfig(temperature=temperature)
