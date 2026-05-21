"""Explore — read-only repo scanner. Cheap and fast (Haiku-class)."""
from repo2rocm.agents.definition import AgentDefinition
from repo2rocm.core.permissions import PermissionMode

_PROMPT = """You are an Explore agent — a focused, read-only scout for the Coordinator.

You have ONLY read tools: Read, Grep, Glob, DockerExec (read-only commands).
You CANNOT edit files, run pip, run apt, change the base image, or spawn
sub-agents. Any attempt will be denied by the permission system.

Your job is to return a TIGHT, FACTUAL answer to the specific question in your prompt.
Do NOT explore beyond what was asked. Do NOT propose fixes. The Coordinator
synthesizes; you observe.

Format:
  - Always cite file paths and (when possible) line numbers.
  - Quote actual file content when the answer requires it.
  - End with a 3-5 line "FINDINGS:" summary the Coordinator can paste into its plan."""

EXPLORE = AgentDefinition(
    name="explore",
    description="Read-only repo scanner. Returns facts, not opinions.",
    allowed_tools=["Read", "Grep", "Glob", "DockerExec"],
    # BYPASS at the mode layer; safety here comes from the read-only allow-list above
    # (no Edit/Write/ApplyDiff/etc). PLAN used to live here belt-and-suspenders style,
    # but PLAN also denies internal tools (EmitPlan, PaperFetch, ...) which broke the
    # whole pipeline. Mode is a coarse switch; allow-list is the real boundary.
    permission_mode=PermissionMode.BYPASS,
    omit_user_context=True,
    max_turns=30,
    max_tokens=4_096,
    system_prompt_template=_PROMPT,
    color="green",
)
