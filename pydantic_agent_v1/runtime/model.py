"""Model resolution — one place that decides which LLM the agents talk to.

``provider:model`` strings (e.g. ``"openai:gpt-5.4-mini"``) are passed straight to
Pydantic AI's ``Agent(model=...)``. The name is resolved from the env so the whole
repo can be pointed at a different model without touching code:

    PYDANTIC_AGENT_V1_MODEL  →  AGENT_V2_OPENAI_MODEL  →  openai:gpt-5.4-mini
"""

from __future__ import annotations

import os

from pydantic_ai.settings import ModelSettings

# ``openai-chat:`` pins the Chat Completions API explicitly (vs the bare ``openai:``
# prefix, which Pydantic AI v2 will repoint at the Responses API).
_DEFAULT_MODEL = "openai-chat:gpt-5.4-mini"


def model_name() -> str:
    name = os.environ.get("PYDANTIC_AGENT_V1_MODEL") or os.environ.get("AGENT_V2_OPENAI_MODEL")
    if not name:
        return _DEFAULT_MODEL
    return name if ":" in name else f"openai-chat:{name}"  # a bare id is assumed OpenAI


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
