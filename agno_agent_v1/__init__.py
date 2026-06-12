"""agno_agent_v1 — the ``agent_v4_1`` shopping assistant, rebuilt on the Agno SDK.

A faithful re-implementation of ``agent_v4_1`` (router orchestrator → context-
isolated sub-agents → deterministic blocks → streaming writer) on top of
``agno.Agent``, written from scratch with no imports from any other package.

Layers:

* :mod:`agno_agent_v1.domain` — pure e-commerce business logic (cart, catalog,
  pricing, serviceability, orders, store). No framework.
* :mod:`agno_agent_v1.agent` — the Agno agent layer. Each agent abstraction has
  its own home: ``prompts`` (voice), ``tools`` (actions), ``skills`` (stateful
  instruction injection), ``builder`` (AgentSpec → ``agno.Agent``), assembled in
  ``agents`` and driven by the streaming ``session``.
"""

__all__ = ["domain", "agent"]
