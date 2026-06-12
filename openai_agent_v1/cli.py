"""A small CLI / scripted harness for ``openai_agent_v1``.

Run a scripted multi-turn conversation (or an interactive REPL) through
:class:`ShoppingSession.run_turn_stream` and pretty-print the live event stream —
routing, tool calls, step summaries, streamed writer tokens, rich blocks, and the
resulting cart state. This is the manual-eval driver used to validate the port.

    uv run python -m openai_agent_v1.cli            # interactive REPL
    uv run python -m openai_agent_v1.cli --demo     # a complex scripted scenario
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from openai_agent_v1.shopping.session import ShoppingSession

# ANSI colors (cheap, no dep)
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
RESET = "\033[0m"


def _fmt_cart(snapshot: dict) -> str:
    cart = snapshot.get("cart", {})
    items = cart.get("items", [])
    item_str = ", ".join(f"{i['id']}×{i['qty']}" for i in items) or "empty"
    bits = [f"step={cart.get('step')}", f"items=[{item_str}]", f"subtotal={cart.get('subtotal')}"]
    if cart.get("grand_total") is not None:
        bits.append(f"total={cart.get('grand_total')}")
    if cart.get("confirmed"):
        bits.append(f"{GREEN}CONFIRMED {cart.get('receipt_id')}{RESET}")
    blockers = cart.get("blockers", [])
    if blockers:
        bits.append(f"{YELLOW}blockers={[b['code'] for b in blockers]}{RESET}")
    return " ".join(bits)


async def run_turn(session: ShoppingSession, text: str, *, show_trace: bool = False) -> None:
    print(f"\n{BOLD}{CYAN}┌─ USER:{RESET} {text}")
    last_snapshot = None
    wrote_bot_prefix = False
    async for ev in session.run_turn_stream(text):
        t = ev["type"]
        if t == "guardrail":
            print(f"{RED}│  ⚠ guardrail[{ev['stage']}] {ev['rule']} → {ev['action']}{RESET}")
        elif t == "router":
            print(f"{DIM}│  → route → {ev['target']}{RESET}")
        elif t == "tool_start":
            args = ", ".join(f"{k}={v!r}" for k, v in (ev.get("args") or {}).items())
            print(f"{DIM}│     ⚙ {ev['name']}({args}){RESET}")
        elif t == "tool_end":
            res = str(ev.get("result", "")).replace("\n", " ⏎ ")
            if len(res) > 110:
                res = res[:110] + "…"
            print(f"{DIM}│       ↳ {res}{RESET}")
        elif t == "skill":
            print(f"{DIM}│     📚 load_skill({ev.get('name')}){RESET}")
        elif t == "step":
            asks = f" asks={ev['asks']}" if ev.get("asks") else ""
            print(f"{MAGENTA}│  ✓ step[{ev['sop']}]: {ev['summary']}{asks}{RESET}")
        elif t == "trace" and show_trace:
            print(f"{DIM}│  🔍 trace[{ev['phase']}] {ev['title']}{RESET}")
        elif t == "token":
            if not wrote_bot_prefix:
                sys.stdout.write(f"{BOLD}{GREEN}│  BOT:{RESET} ")
                wrote_bot_prefix = True
            sys.stdout.write(ev["content"])
            sys.stdout.flush()
        elif t == "bot":
            if not wrote_bot_prefix:
                print(f"{BOLD}{GREEN}│  BOT:{RESET} {ev['content']}", end="")
            blocks = ev.get("blocks") or []
            if blocks:
                kinds = ", ".join(b.get("kind", "?") for b in blocks)
                print(f"\n{DIM}│  ▦ blocks: {kinds}{RESET}", end="")
        elif t == "error":
            print(f"\n{RED}│  ✗ ERROR: {ev['content']}{RESET}", end="")
        elif t == "state":
            last_snapshot = ev["snapshot"]
    if wrote_bot_prefix:
        print()
    if last_snapshot:
        print(f"{DIM}└─ cart: {_fmt_cart(last_snapshot)}{RESET}")


# A complex scenario: browse → reference resolution → backtrack (remove/replace) →
# checkout → unserviceable address (backtrack) → mid-checkout cart edit (stale
# pricing) → payment fix → confirm.
DEMO_SCRIPT = [
    "hi! what can you do?",
    "show me your hoodies and caps",
    "add the green one to my cart",
    "actually I want 2 of those",
    "hmm, on second thought remove the cap and add the black hoodie instead",
    "ok let's check out. I'm Ada Lovelace",
    "ship it to 1 Broadway, Oakland 94607",
    "oh that won't work? use 500 Howard St, San Francisco 94105 then",
    "2 hour delivery please",
    "wait, actually make the hoodie quantity 3 before we pay",
    "pay with card",
    "card token tok_visa_4242",
    "yes, place the order",
]


async def _run_demo(show_trace: bool) -> None:
    session = ShoppingSession(debug=show_trace)
    for text in DEMO_SCRIPT:
        await run_turn(session, text, show_trace=show_trace)


async def _run_repl(show_trace: bool) -> None:
    session = ShoppingSession(debug=show_trace)
    print("Interactive shopping assistant (Ctrl-D to exit).")
    loop = asyncio.get_event_loop()
    while True:
        try:
            text = await loop.run_in_executor(None, input, f"\n{CYAN}you> {RESET}")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.strip():
            await run_turn(session, text, show_trace=show_trace)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="openai_agent_v1 shopping CLI")
    parser.add_argument("--demo", action="store_true", help="run the scripted demo scenario")
    parser.add_argument("--trace", action="store_true", help="show deep-trace event markers")
    parser.add_argument("message", nargs="*", help="a single message to send (one-shot)")
    args = parser.parse_args(argv)

    if args.demo:
        asyncio.run(_run_demo(args.trace))
    elif args.message:
        async def _one() -> None:
            session = ShoppingSession(debug=args.trace)
            await run_turn(session, " ".join(args.message), show_trace=args.trace)

        asyncio.run(_one())
    else:
        asyncio.run(_run_repl(args.trace))


if __name__ == "__main__":
    main()
