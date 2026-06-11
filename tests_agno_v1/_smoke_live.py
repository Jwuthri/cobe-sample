"""Live end-to-end smoke (spends OpenAI tokens). Run: python -m tests_agno_v1._smoke_live"""

from __future__ import annotations

import asyncio
from collections import Counter

from agent_agno_v1.shopping.session import ShoppingSession

TURNS = [
    "show me hoodies",
    "add the black hoodie",
    "my name is Ada Lovelace",
    "ship to 123 Main St, San Francisco, CA 94110",
    "2 hour delivery, and I'll pay cash",
    "yes, place the order",
]


async def main() -> None:
    s = ShoppingSession(user_id="demo", session_id="smoke-1")
    for i, turn in enumerate(TURNS, 1):
        kinds: Counter = Counter()
        routers: list[str] = []
        tools: list[str] = []
        steps: list[str] = []
        token_chunks = 0
        bot = ""
        block_kinds: list[str] = []
        async for ev in s.run_turn_stream(turn):
            kinds[ev["type"]] += 1
            if ev["type"] == "router":
                routers.append(ev["target"])
            elif ev["type"] == "tool_start":
                tools.append(ev["name"])
            elif ev["type"] == "step":
                steps.append(f"{ev['sop']}:{ev['summary']}")
            elif ev["type"] == "token":
                token_chunks += 1
            elif ev["type"] == "bot":
                bot = ev["content"]
                block_kinds = [b["kind"] for b in (ev.get("blocks") or [])]
        print(f"\n=== TURN {i}: {turn!r}")
        print(f"  routers : {routers}")
        print(f"  tools   : {tools}")
        print(f"  steps   : {steps}")
        print(f"  tokens  : {token_chunks} streamed chunks")
        print(f"  blocks  : {block_kinds}")
        print(f"  bot     : {bot[:240]}")
    cart = s.cart_service.cart
    print("\n=== FINAL CART")
    print(f"  step={cart.step.value} confirmed={cart.confirmed} receipt={cart.receipt_id} total={cart.grand_total}")
    assert token_chunks > 0 or bot, "expected streamed tokens or a final bot message"


if __name__ == "__main__":
    asyncio.run(main())
