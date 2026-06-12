"""Reference resolution — the orchestrator's memory of facts across turns.

Sub-agents are context-isolated, so the orchestrator must resolve the user's
references ("the green one", "make it 2") itself. It gets two *domain-agnostic*
sources — never the raw chat:

  * **live** structured state — ``ctx.routing_context()`` (e.g. the current cart);
  * **persisted** per-step recalls — opaque text a sub-agent surfaced last turn
    (``StepResult.recall``), kept keyed by sop in ``routing_notes``.

Both functions are pure helpers over those two sources; the session owns the
persisted ``routing_notes`` dict and passes it in. Nothing here inspects domain
internals, so this generalizes to any tenant.
"""

from __future__ import annotations

from lg_agent.core.context import TurnContext


def build_memo(ctx: TurnContext, routing_notes: dict[str, str]) -> str | None:
    """Assemble the reference-resolution block injected before the last user turn.

    Returns ``None`` when there is nothing to resolve against.
    """
    live = ctx.routing_context()
    blocks = [text for text in live.values() if text]
    blocks += [text for key, text in routing_notes.items() if key not in live and text]
    if not blocks:
        return None
    return (
        "Context for resolving the user's references (authoritative — do NOT invent; "
        "if a reference is ambiguous, delegate to a sub-agent to look it up):\n"
        + "\n".join(blocks)
    )


def absorb_recalls(ctx: TurnContext, routing_notes: dict[str, str]) -> None:
    """Persist each step's ``recall`` snippet (keyed by sop) for future turns."""
    for sr in ctx.step_results:
        if sr.recall:
            routing_notes[sr.sop] = sr.recall


__all__ = ["build_memo", "absorb_recalls"]
