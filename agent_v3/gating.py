"""Skill-based tool-gating — re-implemented as an Agno tool hook.

agent_v2 enforced "you can't call ``set_address`` until you've loaded the
``collect_address`` skill" two ways:
  - ``SkillsMiddleware`` owned a ``load_skill`` tool that appended the
    skill name to ``state["skills_loaded"]``; and
  - every constrained tool called ``_require_skill(runtime, "...")`` and
    refused if its skill wasn't loaded.

In agent_v3 we use **native Agno Skills** for the instruction content
(progressive disclosure via ``get_skill_instructions``) and re-create the
*gating* with a single Agno **tool hook** attached at the agent level.
Verified against agno 2.6.9: agent-level ``tool_hooks`` wrap BOTH our
checkout tools and the native skill tools (``agent/_tools.py``), so one
hook can do both jobs:

  1. When the agent loads a skill (``get_skill_instructions(skill_name)``)
     we record it in ``session_state["skills_loaded"]`` — the agno-native
     analogue of the old ``load_skill`` state update.
  2. When a gated checkout tool is called before its skill is loaded we
     short-circuit with an instructive error (the tool body never runs) —
     the analogue of ``_require_skill``.
"""

from __future__ import annotations

from typing import Any, Callable

# tool name -> the skill that must be loaded first.
# (inverted from each checkout skill's ``unlocks`` list)
# NOTE: native Agno skill names must be lowercase + hyphenated (the spec
# validator rejects underscores), so v2's ``collect_identity`` becomes
# ``collect-identity`` here.
SKILL_FOR_TOOL: dict[str, str] = {
    "set_customer": "collect-identity",
    "set_address": "collect-address",
    "lookup_serviceability": "lookup-serviceability",
    "set_delivery_option": "collect-delivery",
    "quote_shipping": "collect-delivery",
    "compute_tax": "collect-delivery",
    "attach_payment": "collect-payment",
    "confirm_checkout": "collect-payment",
}

# The native Agno Skills tool the agent calls to load a skill's instructions.
SKILL_LOADER_TOOL = "get_skill_instructions"


def _skills_list(run_context: Any) -> list[str]:
    """The shared ``skills_loaded`` list for this run.

    Prefer ``dependencies["skills_loaded"]`` (guaranteed shared by
    reference — agno shallow-copies the deps dict, keeping value refs) so
    loads survive back into the workflow's session_state. Fall back to
    ``session_state`` (used in unit tests that pass a bare run_context).
    """
    deps = getattr(run_context, "dependencies", None) or {}
    if isinstance(deps.get("skills_loaded"), list):
        return deps["skills_loaded"]
    session_state = getattr(run_context, "session_state", None)
    if isinstance(session_state, dict):
        return session_state.setdefault("skills_loaded", [])
    return []


def _loaded_skills(run_context: Any) -> list[str]:
    return _skills_list(run_context)


def _mark_loaded(run_context: Any, skill_name: str) -> None:
    if not skill_name:
        return
    loaded = _skills_list(run_context)
    if skill_name not in loaded:
        loaded.append(skill_name)


def skill_gate_hook(
    function_name: str,
    function_call: Callable[..., Any],
    arguments: dict[str, Any],
    run_context: Any = None,
) -> Any:
    """Agno tool hook enforcing skill-gated checkout tools.

    Attach at the agent level: ``Agent(..., tool_hooks=[skill_gate_hook])``.
    """
    # 1) Record skill loads so gated tools unlock (mirrors load_skill).
    if function_name == SKILL_LOADER_TOOL:
        _mark_loaded(run_context, arguments.get("skill_name", ""))
        return function_call(**arguments)

    # 2) Enforce gating for constrained checkout tools (mirrors _require_skill).
    required = SKILL_FOR_TOOL.get(function_name)
    if required is not None and required not in _loaded_skills(run_context):
        return (
            f"Error: this tool requires the '{required}' skill. "
            f"Call get_skill_instructions('{required}') first."
        )

    return function_call(**arguments)
