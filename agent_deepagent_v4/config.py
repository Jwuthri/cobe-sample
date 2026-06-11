"""Model + environment configuration.

Models are returned in deepagents' ``provider:model`` string form. We default
to OpenAI and reuse the repo's existing env vars so this package runs with the
same key/model as the rest of the project (no new secrets to set):

    orchestrator/workers : AGENT_DEEPAGENT_V4_MODEL → AGENT_V4_OPENAI_MODEL
                           → AGENT_V2_OPENAI_MODEL → gpt-5.4-mini
    writer               : AGENT_DEEPAGENT_V4_WRITER_MODEL → (main model)
"""

from __future__ import annotations

import os
import pathlib

_DEFAULT_MODEL = "gpt-5.4-mini"


def load_env(dotenv_path: str | os.PathLike[str] | None = None) -> None:
    """Best-effort .env loader (the repo doesn't ship python-dotenv).

    Reads ``KEY=VALUE`` lines from the repo root ``.env`` and seeds any var not
    already present in the environment. Safe to call repeatedly.
    """
    path = pathlib.Path(dotenv_path) if dotenv_path else pathlib.Path(__file__).resolve().parents[1] / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _bare_model() -> str:
    return (
        os.environ.get("AGENT_DEEPAGENT_V4_MODEL")
        or os.environ.get("AGENT_V4_OPENAI_MODEL")
        or os.environ.get("AGENT_V2_OPENAI_MODEL")
        or _DEFAULT_MODEL
    )


def _as_provider_string(name: str) -> str:
    """Normalize to ``provider:model``; bare names default to the OpenAI provider."""
    return name if ":" in name else f"openai:{name}"


def main_model() -> str:
    """Model string for the orchestrator and worker subagents."""
    return _as_provider_string(_bare_model())


def writer_model() -> str:
    """Model string for the writer subagent (falls back to the main model)."""
    override = os.environ.get("AGENT_DEEPAGENT_V4_WRITER_MODEL")
    return _as_provider_string(override) if override else main_model()
