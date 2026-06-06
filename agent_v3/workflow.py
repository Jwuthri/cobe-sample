"""Agno Workflow assembly — the v3 replacement for v2's LangGraph ``graph.py``.

    input → Loop( Router[supervisor] → {checkout|product_rec|order_status|finalize} )
              ↓ (loop ends when supervisor routes to ``finalize``)
            compose  (writer → checkout_gate → validator → emit, in one step)
              ↓
            done

The SOP steps are function-step "wrappers": they call the Agno SOP
``Agent`` (sharing this turn's ``session_state`` + ``dependencies`` so the
agent's tools mutate the live cart), then build a structured ``StepResult``
the writer consumes — exactly the v2 wrapper contract.

State lives in the shared ``session_state`` dict; the live cart for the
turn lives in ``dependencies["cart_service"]``. ``run_turn`` (sync) and
``stream_turn`` (yields UI events for the SSE server) are the entry points.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

from agno.workflow import Loop, Router, Step, StepInput, StepOutput, Workflow

from agent_v3.checkout import Cart, CartService
from agent_v3.deps import build_dependencies, get_cart_service
from agent_v3.memory import Store, build_store
from agent_v3.sop_names import SOPName
from agent_v3.sops import (
    build_checkout_agent,
    build_order_status_agent,
    build_product_rec_agent,
)
from agent_v3.state import (
    MAX_VALIDATOR_RETRIES,
    ValidationError,
    fresh_state,
    last_user_message,
    load_cart,
    reset_turn,
    save_cart,
)
from agent_v3.step_result import StepResult
from agent_v3.supervisor import (
    DONE_SENTINEL,
    FINALIZE_STEP,
    MAX_ITERATIONS,
    supervisor_selector,
)
from agent_v3.writer import generate_draft

# ----- module singletons (mirrors v2 graph.py; tests monkeypatch these) -----
_STORE: Store = build_store()
_CHECKOUT_AGENT = build_checkout_agent()
_ORDER_STATUS_AGENT = build_order_status_agent()
_PRODUCT_REC_AGENT = build_product_rec_agent()

_PRODUCT_REC_HISTORY_TURNS = 8
UI_EVENTS_KEY = "_ui_events"


# ============================================================ UI event queue
def _emit_ui(session_state: dict[str, Any], event: dict[str, Any]) -> None:
    """Queue a logic event (step/writer/gate/validator/bot) for the SSE server.

    No-op when the queue isn't present (non-streaming ``run_turn``).
    """
    queue = session_state.get(UI_EVENTS_KEY)
    if isinstance(queue, list):
        queue.append(event)


# ============================================================ run extraction
# Matches catalog tool output: "P-2: Black Hoodie — $49.99 [apparel, hoodie, black]"
_PRODUCT_LINE_RE = re.compile(r"^(P-\d+):\s+(.+?)\s+[—\-]\s+\$(\S+)\s+\[(.+?)\]$")


def _tool_results(resp: Any, names: tuple[str, ...]) -> list[str]:
    """Pull tool result strings (in call order) for the given tool names."""
    out: list[str] = []
    for t in getattr(resp, "tools", None) or []:
        if getattr(t, "tool_name", None) in names:
            result = getattr(t, "result", None)
            if result is not None:
                out.append(str(result))
    return out


def _extract_products(resp: Any) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    seen: set[str] = set()
    for content in _tool_results(resp, ("search_products", "get_product")):
        for line in content.splitlines():
            m = _PRODUCT_LINE_RE.match(line.strip())
            if not m:
                continue
            pid = m.group(1)
            if pid in seen:
                continue
            seen.add(pid)
            products.append(
                {
                    "id": pid,
                    "name": m.group(2),
                    "price": m.group(3),
                    "tags": [t.strip() for t in m.group(4).split(",")],
                }
            )
    return products


def _extract_serviceability(resp: Any) -> dict[str, Any] | None:
    results = _tool_results(resp, ("check_serviceability",))
    for content in reversed(results):
        if content.strip():
            return {"raw": content.strip()}
    return None


def _extract_order(resp: Any) -> dict[str, Any] | None:
    for content in _tool_results(resp, ("get_order_status", "list_recent_orders")):
        c = content.strip()
        if c and "unknown order" not in c.lower():
            return {"raw": c}
    return None


def _run_agent(agent: Any, input_text: str, session_state: dict[str, Any], run_context: Any) -> Any:
    """Invoke a SOP agent, sharing the turn's session_state + dependencies."""
    return agent.run(
        input=input_text,
        session_state=session_state,
        dependencies=getattr(run_context, "dependencies", None),
        user_id=session_state.get("user_id"),
        session_id=session_state.get("session_id"),
    )


def _record_step(session_state: dict[str, Any], sr: StepResult) -> None:
    session_state["step_results"].append(sr.model_dump(mode="json"))
    session_state["active_sop"] = sr.sop.value
    _emit_ui(
        session_state,
        {
            "type": "step",
            "sop": sr.sop.value,
            "summary": sr.summary,
            "asks": list(sr.asks),
            "next_sop": sr.next_sop.value if sr.next_sop else None,
            "details": sr.details,
        },
    )


# ============================================================ SOP steps
def _asks_for_step(step_value: str, cart: Cart) -> list[str]:
    if step_value == "collecting_identity":
        return ["first name", "last name"]
    if step_value == "collecting_address":
        return ["street", "city", "state", "zip code"]
    if step_value == "awaiting_serviceability":
        return ["(internal: serviceability lookup)"]
    if step_value == "collecting_delivery":
        opts = ", ".join(cart.serviceable_options) or "available delivery options"
        return [f"delivery option ({opts})"]
    if step_value == "collecting_payment":
        return ["payment method (card / cash / wallet)", "card_token if paying by card"]
    return []


def checkout_step(step_input: StepInput, session_state: dict[str, Any], run_context: Any) -> StepOutput:
    cs = get_cart_service(run_context)
    _run_agent(_CHECKOUT_AGENT, last_user_message(session_state), session_state, run_context)
    cart = cs.cart

    asks: list[str] = []
    if cart.step.value.startswith("collecting_"):
        asks = _asks_for_step(cart.step.value, cart)
    elif cart.ready_to_confirm() and not cart.confirmed:
        asks = ["explicit yes to place the order"]

    sr = StepResult(
        sop=SOPName.CHECKOUT,
        summary=f"checkout subagent finished at step={cart.step.value}; items={len(cart.items)}",
        asks=asks,
        next_sop=None,
        cart_diff={"step": cart.step.value},
    )
    _record_step(session_state, sr)
    save_cart(session_state, cart)
    return StepOutput(content=sr.summary)


def product_rec_step(step_input: StepInput, session_state: dict[str, Any], run_context: Any) -> StepOutput:
    cs = get_cart_service(run_context)
    items_before = len(cs.cart.items)

    history = session_state.get("messages", [])[-_PRODUCT_REC_HISTORY_TURNS:]
    if history:
        lines = []
        for m in history:
            role = "USER" if m.get("role") == "human" else "ASSISTANT"
            lines.append(f"{role}: {m.get('content', '')}")
        input_text = "Recent conversation:\n" + "\n".join(lines)
    else:
        input_text = last_user_message(session_state)

    resp = _run_agent(_PRODUCT_REC_AGENT, input_text, session_state, run_context)
    products = _extract_products(resp)
    serviceability = _extract_serviceability(resp)

    cart_now = cs.cart
    added_ids = [i.product_id for i in cart_now.items[items_before:]]

    next_sop: SOPName | None = None
    asks: list[str] = []
    details: dict[str, Any] | None = None

    if added_ids:
        summary = f"added {', '.join(added_ids)} to cart"
        next_sop = SOPName.CHECKOUT
        details = {"added": added_ids}
        if products:
            details["products"] = products
    elif serviceability:
        summary = "answered a serviceability question"
        details = {"serviceability": serviceability}
        if products:
            details["products"] = products
    elif products:
        summary = f"catalog returned {len(products)} matching product(s)"
        asks = ["pick a product id (e.g. P-1) to add to your cart"]
        details = {"products": products}
    else:
        summary = "no products matched the user's query"
        asks = ["clarify what you're looking for"]

    sr = StepResult(
        sop=SOPName.PRODUCT_REC,
        summary=summary,
        asks=asks,
        next_sop=next_sop,
        details=details,
        cart_diff={"items": len(cart_now.items)} if added_ids else None,
    )
    _record_step(session_state, sr)
    save_cart(session_state, cart_now)
    return StepOutput(content=sr.summary)


def order_status_step(step_input: StepInput, session_state: dict[str, Any], run_context: Any) -> StepOutput:
    cs = get_cart_service(run_context)
    resp = _run_agent(_ORDER_STATUS_AGENT, last_user_message(session_state), session_state, run_context)
    order_details = _extract_order(resp)
    sr = StepResult(
        sop=SOPName.ORDER_STATUS,
        summary=("looked up order status" if order_details else "could not find a matching order"),
        asks=[] if order_details else ["confirm the order id"],
        next_sop=None,
        details=order_details,
    )
    _record_step(session_state, sr)
    save_cart(session_state, cs.cart)
    return StepOutput(content=sr.summary)


def finalize_step(step_input: StepInput, session_state: dict[str, Any]) -> StepOutput:
    """Terminal loop step — ends the supervisor loop (end_condition watches the content)."""
    session_state["done"] = True
    return StepOutput(content=DONE_SENTINEL)


def _loop_ended(outputs: list[StepOutput]) -> bool:
    """End the supervisor loop once ``finalize`` has run.

    The loop's steps=[Router]; ``end_condition`` receives the Router's
    wrapper StepOutput (content="Router … completed"), with the inner
    selected-step output nested under ``.steps``. So we check both levels
    for the finalize sentinel.
    """
    for o in outputs:
        if getattr(o, "content", None) == DONE_SENTINEL:
            return True
        for sub in getattr(o, "steps", None) or []:
            if getattr(sub, "content", None) == DONE_SENTINEL:
                return True
    return False


# ============================================================ compose (writer+gate+validator+emit)
_PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}|<[A-Z_]+>")
_UNSAFE_RE = re.compile(r"\b(damn|hate you|stupid customer)\b", re.I)
MAX_RESPONSE_CHARS = 2000
_GATE_CLAIM_KEYWORDS = ("confirmed", "placed", "your order", "all set", "thank you")


def _gate_error(session_state: dict[str, Any], cart: Cart, draft: str) -> ValidationError | None:
    """Re-assert Cart.blockers() if the draft claims confirmation (defense-in-depth)."""
    if session_state.get("active_sop") != SOPName.CHECKOUT.value:
        return None
    text = (draft or "").lower()
    claims_done = any(k in text for k in _GATE_CLAIM_KEYWORDS)
    if claims_done and not cart.ready_to_confirm():
        blockers = "; ".join(b.code for b in cart.blockers())
        return ValidationError(
            code="gate", detail=f"model claimed confirm but blockers remain: {blockers}"
        )
    return None


def _validate(draft: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    d = (draft or "").strip()
    if not d:
        errors.append(ValidationError(code="empty", detail="writer produced no text"))
        return errors
    if len(d) > MAX_RESPONSE_CHARS:
        errors.append(ValidationError(code="too_long", detail=f"{len(d)} chars"))
    if _PLACEHOLDER_RE.search(d):
        errors.append(ValidationError(code="placeholder_leak", detail="unfilled template token"))
    if _UNSAFE_RE.search(d):
        errors.append(ValidationError(code="unsafe", detail="safety blocklist hit"))
    return errors


def compose_step(step_input: StepInput, session_state: dict[str, Any], run_context: Any) -> StepOutput:
    """Writer → checkout_gate → validator → emit, collapsed into one step.

    Same checks and retry budget as v2's four graph nodes; the gate, rather
    than bouncing back into the supervisor loop, regenerates the draft with
    a correction (the safety property — never claim 'confirmed' while
    blockers remain — is preserved).
    """
    cart = get_cart_service(run_context).cart
    correction: str | None = None

    draft = generate_draft(session_state, cart, correction)
    _emit_ui(session_state, {"type": "writer", "draft": draft})

    # checkout_gate (defense-in-depth)
    gate_err = _gate_error(session_state, cart, draft)
    if gate_err is not None:
        _emit_ui(session_state, {"type": "gate", "rejected": True, "errors": [gate_err.detail]})
        blockers = "; ".join(b.code for b in cart.blockers())
        correction = (
            "Do NOT claim the order is confirmed or placed — it is not. "
            f"Outstanding blockers: {blockers}. Tell the user what is still needed."
        )
        draft = generate_draft(session_state, cart, correction)
        _emit_ui(session_state, {"type": "writer", "draft": draft})

    # validator (retry budget)
    attempts = 0
    while True:
        errors = _validate(draft)
        if not errors:
            break
        _emit_ui(session_state, {"type": "validator", "errors": [e.code for e in errors]})
        attempts += 1
        if attempts > MAX_VALIDATOR_RETRIES:
            draft = "Sorry, I couldn't produce a clean response. Could you rephrase?"
            break
        draft = generate_draft(session_state, cart, correction)
        _emit_ui(session_state, {"type": "writer", "draft": draft})

    # emit
    session_state["draft_response"] = draft
    session_state["messages"].append({"role": "ai", "content": draft})
    session_state["done"] = True
    _emit_ui(session_state, {"type": "bot", "content": draft})
    return StepOutput(content=draft)


# ============================================================ build
def build_workflow() -> Workflow:
    return Workflow(
        name="agent_v3",
        telemetry=False,
        steps=[
            Loop(
                name="supervise",
                max_iterations=MAX_ITERATIONS + 2,
                end_condition=_loop_ended,
                steps=[
                    Router(
                        name="supervisor",
                        selector=supervisor_selector,
                        choices=[
                            Step(name=SOPName.CHECKOUT.value, executor=checkout_step),
                            Step(name=SOPName.PRODUCT_REC.value, executor=product_rec_step),
                            Step(name=SOPName.ORDER_STATUS.value, executor=order_status_step),
                            Step(name=FINALIZE_STEP, executor=finalize_step),
                        ],
                    )
                ],
            ),
            Step(name="compose", executor=compose_step),
        ],
    )


_WORKFLOW: Workflow | None = None


def _workflow() -> Workflow:
    global _WORKFLOW
    if _WORKFLOW is None:
        _WORKFLOW = build_workflow()
    return _WORKFLOW


# ============================================================ turn entry points
def _prepare_turn(session_state: dict[str, Any], user_msg: str) -> tuple[CartService, dict[str, Any]]:
    session_state.setdefault("messages", []).append({"role": "human", "content": user_msg})
    reset_turn(session_state)
    cart = load_cart(session_state)
    cart_service = CartService(cart)
    deps = build_dependencies(cart_service, _STORE, session_state.setdefault("skills_loaded", []))
    return cart_service, deps


def run_turn(session_state: dict[str, Any], user_msg: str) -> dict[str, Any]:
    """Run one turn to completion (non-streaming). Returns the updated session_state."""
    cart_service, deps = _prepare_turn(session_state, user_msg)
    _workflow().run(
        input=user_msg,
        session_state=session_state,
        dependencies=deps,
        session_id=session_state.get("session_id"),
        user_id=session_state.get("user_id"),
    )
    save_cart(session_state, cart_service.cart)
    return session_state


# ----- Agno stream event -> UI event mapping (for the SSE server) -----
def _router_target(selected: list[str]) -> str:
    if not selected:
        return "writer"
    target = selected[0]
    return "writer" if target == FINALIZE_STEP else target


def _map_native_event(ev: Any, session_state: dict[str, Any]) -> list[dict[str, Any]]:
    name = type(ev).__name__
    out: list[dict[str, Any]] = []
    if name == "RouterExecutionStartedEvent":
        out.append(
            {
                "type": "router",
                "target": _router_target(list(getattr(ev, "selected_steps", []) or [])),
                "iteration": session_state.get("iteration", 0),
            }
        )
    elif name == "StepStartedEvent":
        step_name = getattr(ev, "step_name", None)
        if step_name in (SOPName.CHECKOUT.value, SOPName.PRODUCT_REC.value, SOPName.ORDER_STATUS.value):
            out.append({"type": "agent", "node": f"{step_name}_wrapper"})
    elif name in ("ToolCallStartedEvent",):
        tool = getattr(ev, "tool", None)
        tname = getattr(tool, "tool_name", None)
        targs = getattr(tool, "tool_args", None) or {}
        if tname == "get_skill_instructions":
            out.append({"type": "skill", "name": targs.get("skill_name")})
        elif tname:
            args = {k: v for k, v in targs.items() if k != "run_context"}
            out.append({"type": "tool_start", "name": tname, "args": args})
    elif name in ("ToolCallCompletedEvent",):
        tool = getattr(ev, "tool", None)
        tname = getattr(tool, "tool_name", None)
        if tname and tname != "get_skill_instructions":
            out.append({"type": "tool_end", "name": tname, "result": str(getattr(tool, "result", ""))})
    return out


def _drain_ui(session_state: dict[str, Any]) -> list[dict[str, Any]]:
    queue = session_state.get(UI_EVENTS_KEY)
    if not isinstance(queue, list) or not queue:
        return []
    drained = list(queue)
    queue.clear()
    return drained


def stream_turn(session_state: dict[str, Any], user_msg: str) -> Iterator[dict[str, Any]]:
    """Run one turn, yielding UI logic events (router/agent/skill/tool_*/step/writer/gate/validator/bot).

    The server frames these with ``user`` / ``state`` / ``end`` SSE events.
    """
    cart_service, deps = _prepare_turn(session_state, user_msg)
    session_state[UI_EVENTS_KEY] = []
    try:
        for ev in _workflow().run(
            input=user_msg,
            session_state=session_state,
            dependencies=deps,
            session_id=session_state.get("session_id"),
            user_id=session_state.get("user_id"),
            stream=True,
            stream_events=True,
            stream_executor_events=True,
        ):
            for ui in _map_native_event(ev, session_state):
                yield ui
            for ui in _drain_ui(session_state):
                yield ui
        # flush any trailing queued events
        for ui in _drain_ui(session_state):
            yield ui
    finally:
        save_cart(session_state, cart_service.cart)
        session_state.pop(UI_EVENTS_KEY, None)


__all__ = [
    "build_workflow",
    "run_turn",
    "stream_turn",
    "fresh_state",
]
