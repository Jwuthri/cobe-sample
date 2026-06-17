"""The SSE event vocabulary — the exact wire shapes the web UI consumes.

Every dict here becomes one ``data: {...}`` Server-Sent Event. The frontend
(``web/lib/types.ts`` / ``events.ts``) already understands these types, so this is
the contract that keeps the existing UI working unchanged. Each builder is a tiny
factory so the shapes live in one place instead of being scattered as dict literals.
"""

from __future__ import annotations

from typing import Any

from lg_agent_v3.runtime.step import StepResult


def user(text: str, turn: int) -> dict:
    return {"type": "user", "content": text, "turn": turn}


def state(snapshot: dict) -> dict:
    return {"type": "state", "snapshot": snapshot}


def router(target: str, iteration: int = 0) -> dict:
    return {"type": "router", "target": target, "iteration": iteration}


def agent(node: str) -> dict:
    return {"type": "agent", "node": node}


def tool_start(name: str, args: dict[str, Any]) -> dict:
    return {"type": "tool_start", "name": name, "args": args}


def tool_end(name: str, result: str) -> dict:
    return {"type": "tool_end", "name": name, "result": result}


def step(sr: StepResult) -> dict:
    return {
        "type": "step",
        "sop": sr.sop,
        "summary": sr.summary,
        "asks": list(sr.asks),
        "next_sop": sr.next_sop,
        "details": sr.details,
    }


def guardrail(stage: str, rule: str, action: str) -> dict:
    return {"type": "guardrail", "stage": stage, "rule": rule, "action": action}


def token(text: str) -> dict:
    return {"type": "token", "content": text}


def writer(draft: str, blocks: list[dict]) -> dict:
    return {"type": "writer", "draft": draft, "blocks": blocks}


def bot(content: str, blocks: list[dict]) -> dict:
    return {"type": "bot", "content": content, "blocks": blocks}


def error(message: str) -> dict:
    return {"type": "error", "content": message}


def end() -> dict:
    return {"type": "end"}
