"""Migrator — the write-heavy worker. Full tool set, one per file group.

Permission note: the migrator runs in BYPASS, symmetric with the single-agent
`configuration` flow. The migrator does exactly the same kind of work as the
configuration agent (edit files, DockerExec to install/build, DockerCommit,
Download, ChangeBaseImage, etc.) — just split into focused sub-tasks dispatched
by the coordinator. The Docker sandbox is the safety boundary in both flows;
there is no host-level gating to add value here.

Historical note: this used to be ACCEPT_EDITS, which only auto-allows
Read/Edit/Write/ApplyDiff and turns every non-edit tool (DockerExec,
DockerCommit, Download, ChangeBaseImage, ...) into ASK. The executor in
tools/base.py currently silently passes ASK through, so things appeared to
work — but it's a latent footgun: any future tightening of ASK handling
would silently disable the migrator's install/build path."""
from repo2rocm.agents.definition import AgentDefinition
from repo2rocm.core.permissions import PermissionMode

_PROMPT = """You are a Migrator worker. The Coordinator handed you ONE focused task with
exact file paths and exact intended changes. Execute it. Do NOT explore freely.

You have the full ROCm tool set:
  - Read, Grep, Glob, Edit, Write, ApplyDiff
  - DockerExec, DockerCommit, DockerRollback, ChangeBaseImage, ChangePythonVersion
  - WaitingListAdd, WaitingListAddFile, WaitingListShow, ConflictList* , Download
  - PyPIVersions, DockerHubTags, Fetch, WebSearch
  - EnvVerify

You do NOT have the Agent tool. You cannot spawn sub-workers. If the task is wider
than expected, finish what you can and report the gap in your final message —
the Coordinator will dispatch additional workers.

Workflow:
  1. Read the target file(s) FIRST so the staleness cache is warm.
  2. For each install: PyPIVersions / DockerHubTags FIRST, then Download.
  3. After each successful step, call DockerCommit with a short label.
  4. On a non-recoverable failure, DockerRollback and report.
  5. Only call EnvVerify if the task asked for it; otherwise return to the Coordinator.

Output:
  A single concise summary message: what you did, what failed, the last commit_id."""

MIGRATOR = AgentDefinition(
    name="migrator",
    description="Write-heavy worker. Full tool set; one per file group.",
    allowed_tools=None,  # all tools
    disallowed_tools=["Agent", "SendMessage", "TaskStop"],
    permission_mode=PermissionMode.BYPASS,  # container is the safety boundary
    max_turns=60,
    max_tokens=8_192,
    preload_skills=["cuda_to_rocm_mapping", "py312_compat", "flash_attn_amd_install"],
    system_prompt_template=_PROMPT,
    color="yellow",
)
