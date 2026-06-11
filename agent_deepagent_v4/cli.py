"""Interactive CLI for agent_deepagent_v4.

    uv run python -m agent_deepagent_v4.cli

Type messages to shop. When the order reaches the safe-checkout approval step,
the CLI shows the order summary and asks you to approve/reject before it places.
Commands: /cart  /reset  /quit
"""

from __future__ import annotations

import sys
import uuid

from agent_deepagent_v4.config import load_env
from agent_deepagent_v4.runtime import TurnResult, reset_session, resume_turn, run_turn


def _print_cart(cart: dict) -> None:
    items = ", ".join(f"{i['id']}×{i['qty']}" for i in cart["items"]) or "(empty)"
    print(f"  cart: {items} | step={cart['step']} | total={cart['grand_total']} "
          f"| confirmed={cart['confirmed']}" + (f" | {cart['receipt_id']}" if cart["receipt_id"] else ""))


def _handle_approval(session_id: str, result: TurnResult) -> TurnResult:
    """Drive the human-in-the-loop approval prompt until the turn completes."""
    while result.needs_approval:
        info = result.interrupt or {}
        summary = info.get("summary", {})
        print("\n  ── APPROVAL REQUIRED ──")
        print(f"  {info.get('message', 'Approve this action?')}")
        if summary:
            its = ", ".join(f"{i['id']} {i['name']} ×{i['qty']}" for i in summary.get("items", []))
            print(f"  items: {its}")
            print(f"  total: ${summary.get('grand_total')} | pay: {summary.get('payment_method')} "
                  f"| ship to: {summary.get('ship_to')}")
        ans = input("  Place the order? [y/N]: ").strip().lower()
        decision = {"approved": ans in ("y", "yes")}
        if not decision["approved"]:
            decision["reason"] = "declined at CLI prompt"
        result = resume_turn(session_id, decision)
    return result


def main() -> int:
    load_env()
    session_id = f"cli-{uuid.uuid4().hex[:8]}"
    print("agent_deepagent_v4 — shopping assistant (deepagents)")
    print("Commands: /cart  /reset  /quit\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/reset":
            reset_session(session_id)
            session_id = f"cli-{uuid.uuid4().hex[:8]}"
            print("  (new session)\n")
            continue
        if user == "/cart":
            # A no-op turn just to print the latest cart snapshot.
            from agent_deepagent_v4.runtime import cart_service_for, cart_snapshot

            _print_cart(cart_snapshot(cart_service_for(session_id).cart))
            continue

        result = run_turn(session_id, user)
        result = _handle_approval(session_id, result)
        print(f"\nbot> {result.reply}")
        _print_cart(result.cart)
        print()
    print("bye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
