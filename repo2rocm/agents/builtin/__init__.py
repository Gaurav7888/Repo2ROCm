"""Built-in agent definitions."""
from repo2rocm.agents.builtin.configuration import CONFIGURATION
from repo2rocm.agents.builtin.coordinator import COORDINATOR
from repo2rocm.agents.builtin.explore import EXPLORE
from repo2rocm.agents.builtin.planner import PLANNER
from repo2rocm.agents.builtin.migrator import MIGRATOR
from repo2rocm.agents.builtin.verifier import VERIFIER
from repo2rocm.agents.builtin.paper_research import PAPER_RESEARCH
from repo2rocm.agents.builtin.paper_reproducer import PAPER_REPRODUCER
from repo2rocm.agents.builtin.general_purpose import GENERAL_PURPOSE


def get_builtin_agents() -> dict:
    return {
        "configuration": CONFIGURATION,    # default — single-agent in sandbox
        "coordinator": COORDINATOR,        # optional — multi-agent
        "explore": EXPLORE,
        "planner": PLANNER,
        "migrator": MIGRATOR,
        "verifier": VERIFIER,
        "paper-research": PAPER_RESEARCH,
        "paper-reproducer": PAPER_REPRODUCER,
        "general-purpose": GENERAL_PURPOSE,
    }


__all__ = [
    "get_builtin_agents",
    "CONFIGURATION",
    "COORDINATOR",
    "EXPLORE",
    "PLANNER",
    "MIGRATOR",
    "VERIFIER",
    "PAPER_RESEARCH",
    "PAPER_REPRODUCER",
    "GENERAL_PURPOSE",
]
