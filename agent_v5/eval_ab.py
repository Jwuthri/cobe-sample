"""A/B eval: ``speaking`` (no writer) vs ``router`` (with writer).

Runs ONE scripted multi-turn shopping conversation through both variants, over
several trials (routing has run-to-run variance, so a single sample is not
enough to compare reliability), then scores them on:

  * **Speed**         — wall-clock latency per turn + total (mean over trials).
  * **Cost**          — LLM calls + tokens (supervisor / subagents / writer).
  * **Routing**       — did the right subagent(s) run? Reported as a per-turn
                        stability rate across trials.
  * **Hallucination** — deterministic id checks (no product/order id outside the
                        mock DBs, the live receipt id excepted) + an LLM-judge
                        flag for "claims the order is placed" cross-checked
                        against the real ``cart.confirmed``.
  * **Accuracy/voice**— an LLM judge scores each reply 1–5 for faithfulness to the
                        tool results + appropriateness, ground-truthed on the step
                        results + cart snapshot.

The conversation deliberately includes a mid-checkout interruption ("where's my
order ORD-7?") and a trap question ("is my order placed yet?" before it is) to
test that neither variant reroutes to the wrong agent, loses checkout state, or
falsely claims the order is placed.

NOTE on fairness: the ``speaking`` supervisor runs warm (temp 0.3 — it is the
user-facing voice) and does routing+voice in one prompt; the ``router`` runs cold
(temp 0.0) and defers voice to a temp-0.3 writer. That mirrors how you'd
realistically build each (and v4's own 0.0 classifier / 0.3 writer split). The
coupling of routing to a warm voice model is itself part of what "absorb the
writer" buys/costs — not an artificial confound.

Run:
    python -m agent_v5.eval_ab                 # 3 trials + judge
    python -m agent_v5.eval_ab --trials 2      # fewer trials
    python -m agent_v5.eval_ab --no-judge      # deterministic only (cheaper)
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass, field

from agent_v4.checkout.catalog import CATALOG
from agent_v4.tools.orders_db import ORDERS
from agent_v5.agent import ShoppingAgentV5, TurnResult
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import re

# (user message, set of subagents that SHOULD run; empty = no tool / smalltalk)
SCRIPT: list[tuple[str, set[str]]] = [
    ("hey, what's up?", set()),
    ("what hoodies and caps do you carry?", {"product_rec"}),
    ("add the black hoodie to my cart", {"product_rec"}),
    ("my name is Julien Martin", {"checkout"}),
    ("ship it to 500 Market St, San Francisco, CA 94110", {"checkout"}),
    ("wait — first, where's my order ORD-7?", {"order_status"}),
    ("ok back to it. use the 2 hour delivery", {"checkout"}),
    ("is my order placed yet?", set()),
    ("pay with card, token tok_visa_42", {"checkout"}),
    ("yes, place the order", {"checkout"}),
]

VALID_PRODUCT_IDS = set(CATALOG.keys())
VALID_ORDER_IDS = set(ORDERS.keys())
_PID_RE = re.compile(r"\bP-\d+\b")
_OID_RE = re.compile(r"\b(?:ORD|RCPT)-\d+\b")


class JudgeVerdict(BaseModel):
    faithful: bool = Field(description="Every concrete claim (ids, prices, statuses) is supported by the tool results / cart ground truth.")
    claims_order_placed: bool = Field(description="The message states or implies the order is ALREADY placed/confirmed/paid/shipped.")
    quality: int = Field(description="1-5: how well the reply answers the user's message given the tool results (helpful, right tone, complete).")
    note: str = Field(default="", description="One short sentence.")


_JUDGE_SYSTEM = """You grade ONE assistant reply in a shopping assistant. You are
given the user's message, the tools that ran this turn, the resulting cart
snapshot (ground truth), and the assistant's reply.

Judge ONLY against the provided ground truth:
  - faithful=false if the reply invents a product id/price, an order id/status, or
    any fact not present in the tool results / cart.
  - claims_order_placed=true ONLY if the reply states or implies the order is
    already placed/confirmed/paid/on its way. A request to confirm ("reply yes to
    place the order") is NOT a claim.
  - quality 1-5: does it address the user's message, with the right info and tone?
Return the structured verdict."""


@dataclass
class TurnEval:
    turn: TurnResult
    expected: set[str]
    routing_ok: bool
    bad_product_ids: list[str]
    bad_order_ids: list[str]
    false_confirm: bool
    verdict: JudgeVerdict | None = None


def _deterministic_flags(turn: TurnResult, expected: set[str]) -> tuple[bool, list[str], list[str]]:
    actual = set(turn.sops)
    routing_ok = (actual == expected) if not expected else expected.issubset(actual)
    # The freshly-minted receipt id IS legitimate (it's in the cart snapshot).
    allowed_oids = set(VALID_ORDER_IDS)
    if turn.cart.get("receipt_id"):
        allowed_oids.add(turn.cart["receipt_id"])
    bad_pids = [p for p in _PID_RE.findall(turn.message) if p not in VALID_PRODUCT_IDS]
    bad_oids = [o for o in _OID_RE.findall(turn.message) if o not in allowed_oids]
    return routing_ok, bad_pids, bad_oids


def _judge(turn: TurnResult) -> JudgeVerdict:
    chat = ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_structured_output(JudgeVerdict)
    payload = (
        f"USER: {turn.user_message}\n\n"
        f"TOOLS THAT RAN: {turn.sops or '(none)'}\n"
        f"CART SNAPSHOT (ground truth): {turn.cart}\n\n"
        f"ASSISTANT REPLY:\n{turn.message}"
    )
    return chat.invoke([{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": payload}])


def run_one_trial(variant: str, trial: int, *, judge: bool) -> list[TurnEval]:
    agent = ShoppingAgentV5(variant=variant, session_id=f"eval-{variant}-{trial}")
    evals: list[TurnEval] = []
    for user_text, expected in SCRIPT:
        turn = agent.run_turn(user_text)
        routing_ok, bad_pids, bad_oids = _deterministic_flags(turn, expected)
        verdict = _judge(turn) if judge else None
        false_confirm = bool(verdict and verdict.claims_order_placed and not turn.cart["confirmed"])
        evals.append(TurnEval(turn, expected, routing_ok, bad_pids, bad_oids, false_confirm, verdict))
    return evals


@dataclass
class VariantAgg:
    variant: str
    trials: list[list[TurnEval]] = field(default_factory=list)

    def _per_trial(self, fn) -> list[float]:
        return [fn(t) for t in self.trials]

    # ---- cost / speed (mean over trials) ----
    @property
    def latency(self) -> float:
        return statistics.mean(self._per_trial(lambda t: sum(e.turn.latency_s for e in t)))

    @property
    def calls(self) -> float:
        return statistics.mean(self._per_trial(lambda t: sum(e.turn.usage_total["llm_calls"] for e in t)))

    @property
    def tin(self) -> float:
        return statistics.mean(self._per_trial(lambda t: sum(e.turn.usage_total["input_tokens"] for e in t)))

    @property
    def tout(self) -> float:
        return statistics.mean(self._per_trial(lambda t: sum(e.turn.usage_total["output_tokens"] for e in t)))

    # ---- correctness (counts over all trials*turns) ----
    @property
    def n_turns(self) -> int:
        return sum(len(t) for t in self.trials)

    @property
    def routing_ok(self) -> int:
        return sum(e.routing_ok for t in self.trials for e in t)

    @property
    def false_confirms(self) -> int:
        return sum(e.false_confirm for t in self.trials for e in t)

    @property
    def bad_ids(self) -> int:
        return sum(len(e.bad_product_ids) + len(e.bad_order_ids) for t in self.trials for e in t)

    @property
    def unfaithful(self) -> int:
        return sum(1 for t in self.trials for e in t if e.verdict and not e.verdict.faithful)

    @property
    def completed_order(self) -> int:
        return sum(1 for t in self.trials if t[-1].turn.cart["confirmed"])

    @property
    def avg_quality(self) -> float:
        qs = [e.verdict.quality for t in self.trials for e in t if e.verdict]
        return statistics.mean(qs) if qs else float("nan")

    def routing_rate_per_turn(self) -> list[float]:
        n = len(self.trials)
        return [sum(self.trials[k][i].routing_ok for k in range(n)) / n for i in range(len(SCRIPT))]


def aggregate(variant: str, trials: int, judge: bool) -> VariantAgg:
    agg = VariantAgg(variant=variant)
    for k in range(trials):
        agg.trials.append(run_one_trial(variant, k, judge=judge))
        print(f"  {variant}: trial {k + 1}/{trials} done", flush=True)
    return agg


# ============================================================ report
def _summary(a: VariantAgg, b: VariantAgg, trials: int) -> str:
    def row(label, fa, fb):
        return f"| {label} | {fa} | {fb} |"

    return "\n".join([
        f"## Summary (mean over {trials} trials, {len(SCRIPT)} turns each)",
        "",
        "| metric | speaking (no writer) | router (with writer) |",
        "|--------|----------------------|----------------------|",
        row("total latency (s)", f"{a.latency:.1f}", f"{b.latency:.1f}"),
        row("avg latency / turn (s)", f"{a.latency / len(SCRIPT):.2f}", f"{b.latency / len(SCRIPT):.2f}"),
        row("total LLM calls", f"{a.calls:.0f}", f"{b.calls:.0f}"),
        row("total input tokens", f"{a.tin:,.0f}", f"{b.tin:,.0f}"),
        row("total output tokens", f"{a.tout:,.0f}", f"{b.tout:,.0f}"),
        row("routing correct", f"{a.routing_ok}/{a.n_turns}", f"{b.routing_ok}/{b.n_turns}"),
        row("orders completed", f"{a.completed_order}/{trials}", f"{b.completed_order}/{trials}"),
        row("hallucinated ids", a.bad_ids, b.bad_ids),
        row("false 'order placed' claims", a.false_confirms, b.false_confirms),
        row("unfaithful replies (judge)", a.unfaithful, b.unfaithful),
        row("avg quality (judge 1-5)", f"{a.avg_quality:.2f}", f"{b.avg_quality:.2f}"),
    ])


def _routing_table(a: VariantAgg, b: VariantAgg) -> str:
    ra, rb = a.routing_rate_per_turn(), b.routing_rate_per_turn()
    lines = [
        "## Routing stability (fraction of trials the right subagent ran)",
        "",
        "| # | user message | expected | speaking | router |",
        "|---|--------------|----------|----------|--------|",
    ]
    for i, (user, exp) in enumerate(SCRIPT):
        lines.append(
            f"| {i + 1} | {user[:34]} | {','.join(sorted(exp)) or '—'} | {ra[i]*100:.0f}% | {rb[i]*100:.0f}% |"
        )
    return "\n".join(lines)


def _transcript(agg: VariantAgg) -> str:
    """Trial-0 transcript with per-turn metrics + flags, for qualitative review."""
    t = agg.trials[0]
    lines = [f"### {agg.variant} — trial 0 transcript"]
    for i, e in enumerate(t, 1):
        flags = []
        if not e.routing_ok:
            flags.append("MISROUTE")
        if e.bad_product_ids or e.bad_order_ids:
            flags.append("bad-id:" + ",".join(e.bad_product_ids + e.bad_order_ids))
        if e.false_confirm:
            flags.append("FALSE-CONFIRM")
        if e.verdict and not e.verdict.faithful:
            flags.append("unfaithful")
        q = f"q{e.verdict.quality}" if e.verdict else ""
        meta = f"[ran={','.join(e.turn.sops) or '—'} calls={e.turn.usage_total['llm_calls']} {e.turn.latency_s:.1f}s {q} {' '.join(flags)}]"
        lines.append(f"\n**{i}. user:** {e.turn.user_message}\n**bot:** {e.turn.message}\n`{meta}`")
    return "\n".join(lines)


def main() -> None:
    judge = "--no-judge" not in sys.argv
    trials = 3
    if "--trials" in sys.argv:
        trials = int(sys.argv[sys.argv.index("--trials") + 1])

    print(f"A/B eval: {trials} trials/variant, judge={'on' if judge else 'off'}\n", flush=True)
    a = aggregate("speaking", trials, judge)
    b = aggregate("router", trials, judge)

    report = "\n\n".join([
        "# agent_v5 A/B — speaking (no writer) vs router (with writer)",
        _summary(a, b, trials),
        _routing_table(a, b),
        "## Transcripts (trial 0)",
        _transcript(a),
        _transcript(b),
    ])
    out = "agent_v5/EVAL_RESULT.md"
    with open(out, "w") as f:
        f.write(report + "\n")
    print("\n" + _summary(a, b, trials))
    print("\n" + _routing_table(a, b))
    print(f"\nWrote full report (with transcripts) to {out}")


if __name__ == "__main__":
    main()
