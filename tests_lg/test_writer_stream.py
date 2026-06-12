"""Writer token streaming + retry-on-empty (no real model).

The happy path uses a real ``create_agent`` + ``GenericFakeChatModel`` (which
streams). The empty-reply cases use a tiny ``_FakeWriter`` because the fake chat
model can't emit a genuinely empty stream (it raises) — a real model can.
"""

from __future__ import annotations

import asyncio

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk

from lg_agent.shopping.session import ShoppingSession


class _FakeWriter:
    """astream that replays a scripted list of text deltas per attempt."""

    def __init__(self, attempts: list[list[str]]):
        self.attempts = attempts
        self.n = 0

    async def astream(self, _input, stream_mode=None):
        script = self.attempts[self.n]
        self.n += 1
        for delta in script:
            yield (AIMessageChunk(content=delta), {"langgraph_node": "model"})


def _collect_stream(session, payload="payload"):
    async def go():
        return [ev async for ev in session._stream_writer(payload)]

    return asyncio.run(go())


def test_writer_streams_tokens_joining_to_full_text():
    writer = create_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="Hello world, here you go")])),
        tools=[],
        system_prompt="w",
    )
    session = ShoppingSession(orchestrator=object(), writer=writer)
    evs = _collect_stream(session)
    tokens = [e["content"] for e in evs if e["type"] == "token"]
    final = next(e for e in evs if e["type"] == "_final")["content"]
    assert len(tokens) >= 2  # streamed in pieces (don't assert exact count)
    assert "".join(tokens) == "Hello world, here you go"
    assert final == "Hello world, here you go"


def test_writer_retries_once_on_empty():
    session = ShoppingSession(
        orchestrator=object(), writer=_FakeWriter([[], ["Recovered", " reply"]])
    )
    evs = _collect_stream(session)
    tokens = [e["content"] for e in evs if e["type"] == "token"]
    final = next(e for e in evs if e["type"] == "_final")["content"]
    assert "".join(tokens) == "Recovered reply"  # only attempt 2 produced tokens
    assert final == "Recovered reply"


def test_writer_empty_twice_yields_empty_final():
    session = ShoppingSession(orchestrator=object(), writer=_FakeWriter([[], []]))
    evs = _collect_stream(session)
    assert [e for e in evs if e["type"] == "token"] == []
    assert next(e for e in evs if e["type"] == "_final")["content"] == ""
