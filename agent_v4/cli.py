"""Rich split-screen TUI.

Left panel = live cart state. Right panel = streaming event log.

Streams the LangGraph turn via ``graph.stream(stream_mode=["updates", "custom"])``:
  - ``updates`` mode tells us which outer node fired and what it changed.
  - ``custom`` mode delivers events written by middleware (e.g. tool_start/end
    from ``log_tool_calls``).

Run:
    python -m agent_v4.cli
"""

from __future__ import annotations

import os
import sys
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Deque

from agent_v4.graph import build_graph
from agent_v4.state import AgentState
from langchain_core.messages import AIMessage, HumanMessage
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

_EVENT_STYLES = {
    "USER": "cyan bold",
    "ROUTER": "blue",
    "AGENT": "magenta",
    "SKILL": "yellow",
    "TOOL": "green",
    "CART": "white",
    "STEP": "orange3",
    "GATE": "red",
    "HITL": "red bold",
    "WRITER": "cyan dim",
    "BOT": "cyan",
    "VALIDATOR": "yellow dim",
}


class EventLog:
    def __init__(self, capacity: int = 30) -> None:
        self.events: Deque[Text] = deque(maxlen=capacity)

    def add(self, kind: str, body: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        style = _EVENT_STYLES.get(kind, "white")
        line = Text.assemble(
            (f"{ts} ", "dim"),
            (f"{kind:9s}", style),
            " ",
            (body, ""),
        )
        self.events.append(line)

    def render(self) -> Panel:
        if not self.events:
            return Panel(Text("(no events yet)", style="dim"), title="Events", border_style="dim")
        body = Group(*self.events)
        return Panel(body, title="Events", border_style="blue")


def render_cart_panel(state: AgentState) -> Panel:
    cart = state.cart
    t = Table.grid(padding=(0, 1))
    t.add_column(justify="right", style="dim")
    t.add_column()
    t.add_row("user", state.user_id)
    t.add_row("session", state.session_id)
    t.add_row("active_sop", state.active_sop.value if state.active_sop else "-")
    t.add_row("step", cart.step.value)
    t.add_row("skills", ", ".join(state.skills_loaded) or "-")
    t.add_row("", "")
    t.add_row(
        "customer", f"{cart.customer.first_name or '-'} {cart.customer.last_name or ''}".strip()
    )
    addr = cart.address
    addr_line = (
        f"{addr.street or '-'}, {addr.city or '-'} {addr.zip_code or '-'} {addr.country}"
        if addr.is_complete()
        else "-"
    )
    t.add_row("address", addr_line)
    t.add_row(
        "serviceable",
        (
            "yes (" + ", ".join(cart.serviceable_options) + ")"
            if cart.serviceable
            else ("no" if cart.serviceable is False else "?")
        ),
    )
    t.add_row("", "")
    if cart.items:
        items_str = "\n".join(
            f"  {i.product_id} {i.name} ×{i.quantity}  ${i.line_total:.2f}" for i in cart.items
        )
    else:
        items_str = "  (empty)"
    t.add_row("items", items_str)
    t.add_row("subtotal", f"${cart.subtotal:.2f}")
    t.add_row("delivery", cart.delivery_option or "-")
    t.add_row(
        "shipping",
        (
            f"${cart.shipping.cost:.2f} ({cart.shipping.eta_hours}h)"
            if cart.shipping_is_fresh()
            else "stale/missing"
        ),
    )
    t.add_row("tax", f"${cart.tax.amount:.2f}" if cart.tax_is_fresh() else "stale/missing")
    if cart.promo:
        t.add_row("promo", f"{cart.promo.code} -${cart.promo.discount:.2f}")
    t.add_row(
        "payment", f"{cart.payment_method or '-'} ({'tok set' if cart.card_token else 'no tok'})"
    )
    gt = cart.grand_total
    t.add_row("grand total", f"${gt:.2f}" if gt is not None else "-")
    bs = cart.blockers()
    if bs:
        blocker_lines = "\n".join(f"  - {b.code}" for b in bs)
        t.add_row("blockers", Text(blocker_lines, style="red"))
    else:
        t.add_row("blockers", Text("none ✓", style="green"))
    return Panel(t, title="Cart (live)", border_style="green")


def render_layout(state: AgentState, log: EventLog, footer: str) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )
    layout["main"].split_row(
        Layout(render_cart_panel(state), name="cart", ratio=1),
        Layout(log.render(), name="events", ratio=1),
    )
    layout["footer"].update(Panel(Align.left(footer or " "), border_style="dim"))
    return layout


def _process_update_event(node: str, update: dict, log: EventLog, state: AgentState) -> AgentState:
    """Render one outer-node update and return the new state."""
    if not isinstance(update, dict):
        return state
    if node == "supervisor":
        sop = update.get("active_sop")
        if sop:
            log.add(
                "ROUTER",
                f"→ {sop.value if hasattr(sop, 'value') else sop} (iter {update.get('iteration', '?')})",
            )
        elif update.get("iteration") == 0 and "active_sop" not in update:
            # Supervisor decided we're done; routing to writer.
            log.add("ROUTER", "→ writer (done)")
    elif node.endswith("_wrapper"):
        log.add("AGENT", f"{node} finished")
        for sr in update.get("step_results", []) or []:
            asks = f" asks={sr.asks}" if sr.asks else ""
            nxt = f" → {sr.next_sop}" if sr.next_sop else ""
            log.add("STEP", f"{sr.sop}: {sr.summary}{asks}{nxt}")
    elif node == "writer":
        draft = update.get("draft_response", "")
        if draft:
            short = draft.replace("\n", " ⏎ ")
            if len(short) > 200:
                short = short[:200] + "…"
            log.add("WRITER", short)
    elif node == "validator":
        if update.get("validation_errors"):
            log.add("VALIDATOR", "; ".join(e.code for e in update["validation_errors"]))
    elif node == "emit":
        msgs = update.get("messages", [])
        for m in msgs:
            if isinstance(m, AIMessage):
                log.add("BOT", str(m.content))
    # Merge update into state for the live panel.
    new_state = state.model_copy(update={k: v for k, v in update.items() if k != "messages"})
    return new_state


def _process_custom_event(payload: dict, log: EventLog) -> None:
    """Render a custom stream event from middleware (e.g. log_tool_calls)."""
    ev = payload.get("event")
    if ev == "tool_start":
        name = payload.get("tool")
        if name == "load_skill":
            log.add("SKILL", f"load → {payload.get('args', {}).get('skill_name')}")
        else:
            args = payload.get("args", {})
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "tool_call_id")
            log.add("TOOL", f"{name}({args_str})")
    elif ev == "tool_end":
        # Short result line, dim
        result = payload.get("result", "")
        if result:
            short = result.replace("\n", " ⏎ ")
            if len(short) > 120:
                short = short[:120] + "…"
            log.add("TOOL", f"  → {short}")


def run_cli() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is required. Export it before running.")

    console = Console()
    graph = build_graph()
    state = AgentState(user_id="demo", session_id=f"sess-{uuid.uuid4().hex[:8]}")
    log = EventLog(capacity=40)
    footer = "Type a message, or 'quit' to exit. Reply 'yes' to confirm orders."

    console.print(
        Panel(
            "agent_v4 multi-agent demo — supervisor → subagents → cart gate → emit",
            border_style="magenta",
        )
    )

    while True:
        try:
            user_input = Prompt.ask("[cyan bold]you[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            return

        log.add("USER", user_input)

        # Confirmation is now prompt-gated: the user's "yes" / "y" /
        # "confirm" runs through the normal turn flow and the checkout
        # sub-agent decides whether to call confirm_checkout.
        state = state.model_copy(
            update={
                "messages": state.messages + [HumanMessage(content=user_input)],
                "draft_response": None,
                "draft_blocks": [],
                "validation_errors": [],
                "response_attempts": 0,
                "done": False,
            }
        )

        with Live(
            render_layout(state, log, footer),
            console=console,
            refresh_per_second=8,
            screen=False,
        ) as live:
            try:
                for chunk in graph.stream(state, stream_mode=["updates", "custom"]):
                    mode, payload = chunk
                    if mode == "custom":
                        _process_custom_event(payload, log)
                    elif mode == "updates":
                        for node, update in payload.items():
                            state = _process_update_event(node, update, log, state)
                    live.update(render_layout(state, log, footer))
            except Exception as e:  # noqa: BLE001
                log.add("VALIDATOR", f"error: {e}")
                live.update(render_layout(state, log, footer))

        # Persist messages from the run.
        # `graph.stream` doesn't return final state — re-invoke would be expensive.
        # Instead, append the last BOT event we saw.


if __name__ == "__main__":
    run_cli()
