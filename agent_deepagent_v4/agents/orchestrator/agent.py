"""Build the orchestrator (the main deep agent) from the subagent specs.

This is where the whole topology is assembled:

    create_deep_agent(orchestrator)
      ├─ subagents: product-agent, checkout-agent, order-status-agent, writer-agent
      ├─ middleware: ResponseValidatorMiddleware  (non-empty-reply net)
      ├─ context_schema: ShopContext              (shared cart across subagents)
      ├─ backend: FilesystemBackend(skills/)       (lets subagents load SKILL.md)
      └─ store / checkpointer                      (long-term memory + HITL resume)
"""

from __future__ import annotations

import pathlib
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

from agent_deepagent_v4.agents.checkout.agent import build_checkout_subagent
from agent_deepagent_v4.agents.order_status.agent import build_order_status_subagent
from agent_deepagent_v4.agents.orchestrator.prompt import ORCHESTRATOR_PROMPT
from agent_deepagent_v4.agents.product_rec.agent import build_product_rec_subagent
from agent_deepagent_v4.agents.writer.agent import build_writer_subagent
from agent_deepagent_v4.config import main_model
from agent_deepagent_v4.context import ShopContext
from agent_deepagent_v4.validator import ResponseValidatorMiddleware

# The deepagents filesystem backend is rooted at the package ``skills/`` dir, so a
# subagent skill source like "checkout" resolves to skills/checkout/*/SKILL.md.
SKILLS_ROOT = str(pathlib.Path(__file__).resolve().parents[1].parent / "skills")


def build_orchestrator(*, checkpointer: Any | None = None, store: Any | None = None):
    """Compile the orchestrator deep agent.

    Args:
        checkpointer: persists message history per thread and enables interrupt
            resume. Required for the HITL safe-checkout flow (the checkout agent's
            ``confirm_checkout`` tool pauses via ``interrupt()``).
        store: long-term memory store reachable from tools via ``runtime.store``.
    """
    subagents = [
        build_product_rec_subagent(),
        build_checkout_subagent(),
        build_order_status_subagent(),
        build_writer_subagent(),
    ]
    return create_deep_agent(
        model=main_model(),
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=subagents,
        middleware=[ResponseValidatorMiddleware()],
        context_schema=ShopContext,
        # virtual_mode=False keeps plain on-disk path semantics for the trusted,
        # read-only skills directory (set explicitly to avoid a default-change warning).
        backend=FilesystemBackend(root_dir=SKILLS_ROOT, virtual_mode=False),
        checkpointer=checkpointer,
        store=store,
        name="orchestrator",
    )
