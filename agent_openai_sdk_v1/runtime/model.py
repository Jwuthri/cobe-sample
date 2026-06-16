"""Model resolution — one place that decides which LLM the agents talk to.

A bare model id (e.g. ``"gpt-5.4-mini"``) is passed straight to the OpenAI Agents
SDK's ``Agent(model=...)`` (it goes via the Responses API). The name is resolved
from the env so the whole repo can be pointed at a different model without code:

    AGENT_OPENAI_SDK_V1_MODEL  →  AGENT_V2_OPENAI_MODEL  →  gpt-5.4-mini
"""

from __future__ import annotations

import os

from agents.model_settings import ModelSettings

_DEFAULT_MODEL = "gpt-5.4-mini"


def model_name() -> str:
    return (
        os.environ.get("AGENT_OPENAI_SDK_V1_MODEL")
        or os.environ.get("AGENT_V2_OPENAI_MODEL")
        or _DEFAULT_MODEL
    )


MODEL_NAME = model_name()


def settings(temperature: float, *, parallel_tool_calls: bool | None = None) -> ModelSettings:
    """Build ``ModelSettings`` for an agent.

    ``parallel_tool_calls=False`` is set on the orchestrator so a compound user
    message routes ONE sub-agent per model step — keeping the event stream ordered
    and the shared cart free of concurrent mutation.
    """
    kwargs: dict = {"temperature": temperature}
    if parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = parallel_tool_calls
    return ModelSettings(**kwargs)
