"""Skill sources for the checkout agent.

These are paths (relative to the deepagents filesystem backend root, i.e. the
package ``skills/`` directory) that the framework's ``SkillsMiddleware`` scans.
At startup it injects each skill's name + description into the agent's prompt;
the full ``SKILL.md`` body is read on demand (progressive disclosure). The
actual ``SKILL.md`` files live under ``agent_deepagent_v4/skills/checkout/``.
"""

# A "source" is a directory that contains one subdirectory per skill, each with
# a SKILL.md. ``checkout`` holds the checkout-flow and payments skills.
CHECKOUT_SKILL_SOURCES = ["checkout"]
