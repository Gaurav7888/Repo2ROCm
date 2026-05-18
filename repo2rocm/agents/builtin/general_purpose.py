"""General-purpose fallback agent — used when subagent_type is unrecognized."""
from repo2rocm.agents.definition import AgentDefinition
from repo2rocm.core.permissions import PermissionMode

_PROMPT = """You are a general-purpose worker. Use the available tools to complete the
task fully. Don't gold-plate; don't leave things half-done. Cannot spawn sub-agents."""

GENERAL_PURPOSE = AgentDefinition(
    name="general-purpose",
    description="Default worker. Full tools minus Agent. Use when no specialist fits.",
    allowed_tools=None,
    disallowed_tools=["Agent"],
    permission_mode=PermissionMode.ACCEPT_EDITS,
    max_turns=50,
    max_tokens=8_192,
    system_prompt_template=_PROMPT,
    color="white",
)
