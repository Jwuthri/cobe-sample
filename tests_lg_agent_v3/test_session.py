"""End-to-end pipeline tests driven by scripted fake models (no real LLM).

These exercise the whole streaming turn — guardrails → orchestrator (delegation) →
worker tools → deterministic blocks → streamed writer — and the wire events the UI
consumes, all offline and deterministically.
"""

from __future__ import annotations

from tests_lg_agent_v3.conftest import call_then_done, say, sequence


def _kinds(events, kind):
    return [e for e in events if e["type"] == kind]


def test_browse_and_add_routes_to_product_rec(make_session):
    s = make_session(
        orchestrator=call_then_done(("product_rec", {"query": "add P-2 to the cart"})),
        product_rec=call_then_done(("add_item", {"product_id": "P-2"})),
        writer=say("Added the Black Hoodie to your cart."),
    )
    out = s.run_turn("add the black hoodie")
    events = out["events"]

    # routing + tool + step events surfaced
    assert [r["target"] for r in _kinds(events, "router")] == ["product_rec", "writer"]
    assert any(t["name"] == "add_item" for t in _kinds(events, "tool_start"))
    assert _kinds(events, "step")[0]["sop"] == "product_rec"

    # the cart actually mutated, and the reply streamed token-by-token
    assert [i["id"] for i in s.snapshot()["cart"]["items"]] == ["P-2"]
    assert out["tokens"], "writer should have streamed at least one token"
    assert out["message"] == "Added the Black Hoodie to your cart."

    # a deterministic product_reco block was attached
    assert any(b["kind"] == "product_reco" for b in out["blocks"])


def test_smalltalk_calls_no_worker(make_session):
    s = make_session(
        orchestrator=say("DONE"),  # routes nothing
        writer=say("Hi! I can help you find products and place orders."),
    )
    out = s.run_turn("hello there")
    assert [r["target"] for r in _kinds(out["events"], "router")] == ["writer"]
    assert not _kinds(out["events"], "step")
    assert out["blocks"] == []
    assert out["message"].startswith("Hi!")


def test_checkout_captures_address_and_advances(make_session):
    s = make_session(
        orchestrator=call_then_done(("checkout", {"query": "1 Market St, San Francisco, 94105"})),
        checkout=sequence(
            ("set_address", {"street": "1 Market St", "city": "San Francisco", "zip_code": "94105"}),
            ("lookup_serviceability", {}),
        ),
        writer=say("Thanks, I've got your address."),
    )
    s.cart_service.add_item("P-2", 1)
    s.cart_service.set_customer("Ada", "Lovelace")  # now at collecting_address

    out = s.run_turn("1 Market St, San Francisco, 94105")

    cart = s.snapshot()["cart"]
    assert cart["address"]["zip_code"] == "94105"
    assert cart["serviceable"] is True
    assert cart["step"] == "collecting_delivery"
    # a checkout block reflects the live cart
    assert any(b["kind"] == "checkout" for b in out["blocks"])


def test_empty_cart_hides_checkout_tool(make_session):
    """The empty-cart guard (hide_tool middleware) removes checkout from the router's tools."""
    from langchain_core.messages import AIMessage

    from tests_lg_agent_v3.conftest import ToolCallingFake

    seen: dict[str, list[str]] = {}

    class ToolSpyFake(ToolCallingFake):
        def bind_tools(self, tools, **kwargs):
            seen["tools"] = [getattr(t, "name", None) for t in tools]
            return self

    spy = ToolSpyFake(messages=iter([AIMessage(content="DONE")]))
    s = make_session(orchestrator=spy, writer=say("Hi."))  # empty cart
    s.run_turn("hello")
    assert "checkout" not in seen["tools"]
    assert "product_rec" in seen["tools"]


def test_routing_memo_is_appended_in_the_cacheable_tail(make_session):
    """The memo is APPENDED last (prefix = system+tools+history stays cacheable)."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from tests_lg_agent_v3.conftest import ToolCallingFake

    captured: list[list] = []

    class Recorder(ToolCallingFake):
        # the turn-graph streams the orchestrator model → capture in _stream, not _generate
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            captured.append(list(messages))
            yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    s = make_session(orchestrator=Recorder(messages=iter([AIMessage(content="DONE")])), writer=say("hi"))
    s.cart_service.add_item("P-2", 1)  # non-empty cart → build_memo emits content
    s.run_turn("anything")

    msgs = captured[0]
    assert isinstance(msgs[0], SystemMessage)  # static ROUTER_PROMPT still leads the prefix
    assert isinstance(msgs[-1], SystemMessage) and "Current cart" in msgs[-1].content  # memo is the tail
    assert isinstance(msgs[-2], HumanMessage)  # the new user turn precedes the memo


def test_dynamic_note_never_splits_a_tool_call_pair():
    """A dynamic note must NOT land between an AIMessage(tool_calls) and its ToolMessage.

    That adjacency split is exactly what the provider rejects with a 400 ("tool_call_ids
    did not have response messages"). Exercised through a worker mid-tool-loop, where the
    cart goes non-empty so the cart-note actually renders.
    """
    import asyncio

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    from lg_agent_v3.agents import product_rec as pr
    from lg_agent_v3.domain import CartService, MemoryStore
    from lg_agent_v3.runtime import ShoppingDeps
    from tests_lg_agent_v3.conftest import ToolCallingFake

    calls: list[list] = []

    class Recorder(ToolCallingFake):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            calls.append(list(messages))
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    fake = Recorder(
        messages=iter(
            [AIMessage(content="", tool_calls=[{"name": "add_item", "args": {"product_id": "P-2"}, "id": "c1"}]),
             AIMessage(content="DONE")]
        )
    )
    agent = pr.build(fake)
    deps = ShoppingDeps(cart_service=CartService(), store=MemoryStore())
    asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="add P-2")]}, context=deps))

    second_call = calls[1]  # the call AFTER the tool ran — cart now non-empty → note renders
    # every AIMessage with tool_calls is immediately followed by ToolMessage(s), no System between
    for i, m in enumerate(second_call):
        if isinstance(m, AIMessage) and m.tool_calls:
            assert isinstance(second_call[i + 1], ToolMessage), "a note split a tool-call pair"
    assert isinstance(second_call[-1], SystemMessage) and "Current cart" in second_call[-1].content


def test_cart_mutating_workers_disable_parallel_tool_calls():
    """Cart mutators bind parallel_tool_calls=False (serial); read-only order_status doesn't."""
    import asyncio

    from langchain_core.messages import AIMessage

    from lg_agent_v3.agents import checkout as co
    from lg_agent_v3.agents import order_status as os_
    from lg_agent_v3.agents import product_rec as pr
    from lg_agent_v3.domain import CartService, MemoryStore
    from lg_agent_v3.runtime import ShoppingDeps
    from tests_lg_agent_v3.conftest import ToolCallingFake

    def bind_kwargs(builder):
        seen: dict = {}

        class Spy(ToolCallingFake):
            def bind_tools(self, tools, **kw):
                seen.update(kw)
                return self

        agent = builder(Spy(messages=iter([AIMessage(content="DONE")])))
        deps = ShoppingDeps(cart_service=CartService(), store=MemoryStore())
        asyncio.run(agent.ainvoke({"messages": [AIMessage(content="x")]}, context=deps))
        return seen

    assert bind_kwargs(pr.build).get("parallel_tool_calls") is False
    assert bind_kwargs(co.build).get("parallel_tool_calls") is False
    assert "parallel_tool_calls" not in bind_kwargs(os_.build)  # reads may parallelize


def test_orchestrator_guardrail_blocks_and_writer_delivers_refusal():
    """An orchestrator before_agent block → the writer delivers the refusal verbatim."""
    from lg_agent_v3 import ShoppingSession
    from lg_agent_v3.agents.orchestrator import build_orchestrator
    from lg_agent_v3.runtime.guardrails import GuardrailSpec
    from tests_lg_agent_v3.conftest import say

    g = [GuardrailSpec(type="blocklist", action="block", on_input=True,
                       message="I can't help with that.", params={"phrases": ["forbidden"]})]
    # the orchestrator/writer models are never called: the guardrail short-circuits the
    # orchestrator before routing, and a refusal is delivered verbatim (no writer model).
    s = ShoppingSession(orchestrator_agent=build_orchestrator(model=say("DONE"), guardrails=g),
                        writer_agent=say("unused"))
    out = s.run_turn("this is forbidden")

    assert out["message"] == "I can't help with that."
    assert out["blocks"] == []
    guards = [e for e in out["events"] if e["type"] == "guardrail"]
    assert guards and guards[0]["rule"] == "blocklist" and guards[0]["stage"] == "orchestrator:input"
    # the refusal reached the client as a token too
    assert "".join(out["tokens"]) == "I can't help with that."


def test_subagent_guardrail_block_comes_back_as_a_flagged_refusal(make_session):
    """A blocked sub-agent returns a flagged guardrail step → writer relays it verbatim."""
    from lg_agent_v3.runtime.guardrails import GuardrailSpec
    from tests_lg_agent_v3.conftest import call_then_done, say

    s = make_session(
        orchestrator=call_then_done(("product_rec", {"query": "add P-2"})),
        product_rec=call_then_done(("add_item", {"product_id": "P-2"})),
        product_rec_guardrails=[GuardrailSpec(type="blocklist", action="block", on_input=True,
                                              message="That product can't be sold here.",
                                              params={"phrases": ["P-2"]})],
        writer=say("unused"),
    )
    out = s.run_turn("add P-2")
    assert out["message"] == "That product can't be sold here."
    assert "".join(out["tokens"]) == "That product can't be sold here."
