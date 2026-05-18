"""AgentDefinition — data, not code.

Each builtin/user/plugin agent is described by one of these objects. The 15-step
lifecycle (`agents/lifecycle.py`) interprets a definition uniformly: the agent
type is encoded in *configuration*, not in *control flow*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from repo2rocm.core.permissions import PermissionMode


@dataclass
class AgentDefinition:
    name: str
    description: str

    # Tools
    allowed_tools: list[str] | None = None   # None = all
    disallowed_tools: list[str] = field(default_factory=list)

    # Model
    model: str | None = None  # None = inherit parent
    max_tokens: int = 8_192
    max_turns: int = 60

    # Permissions
    permission_mode: PermissionMode = PermissionMode.DEFAULT

    # Context
    omit_user_context: bool = False  # if True, strip per-project memory injection

    # System prompt
    system_prompt_template: str = ""
    # OPTIONAL builder that gets the runtime context and produces the final system prompt
    system_prompt_builder: Callable[..., str] | None = None

    # Skill preloading (frontmatter names to load into the conversation as user messages)
    preload_skills: list[str] = field(default_factory=list)

    # Lifecycle hooks declared by the agent (frontmatter style)
    hooks: dict[str, list[dict]] = field(default_factory=dict)

    # Run mode
    background: bool = False  # always-async like Verifier

    # MCP servers to attach (in addition to parent's)
    mcp_servers: list[str] = field(default_factory=list)

    # Color (UI)
    color: str = "white"

    def with_(self, **kwargs: Any) -> AgentDefinition:
        from dataclasses import replace

        return replace(self, **kwargs)
