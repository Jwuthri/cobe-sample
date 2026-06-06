"""Runtime context — typed config threaded into every sub-agent invoke.

LangChain v1 distinguishes ``state`` (mutable conversation data) from
``context`` (static-per-invocation config like user_id, API handles).
We pass a ``RuntimeContext`` via ``agent.invoke(..., context=ctx)`` and
tools read it through ``ToolRuntime[RuntimeContext]``.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_v2.checkout.service import CartService


@dataclass
class RuntimeContext:
    """Static-per-invocation config for sub-agents.

    Fields:
      user_id: identifies the user; used as the long-term memory key.
      session_id: identifies the conversation; used as the checkpointer
        thread_id so HITL interrupts survive across turns.
      cart_service: handle to the live CartService for THIS turn. Tools
        mutate ``cart_service.cart`` directly. The outer wrapper reads
        the mutated cart back into AgentState after the subagent returns.
    """

    user_id: str
    session_id: str
    cart_service: CartService
