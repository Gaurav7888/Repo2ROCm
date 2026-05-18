# Tools

Every tool in Repo2ROCm v2 is a `BaseTool` subclass. The class declares its name,
description, input Pydantic model, and overrides the five-method protocol.

## Authoring a new tool

```python
# repo2rocm/tools/myteam/cool_tool.py
from typing import ClassVar
from pydantic import BaseModel
from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class CoolInput(BaseModel):
    path: str
    flag: bool = False


class CoolOutput(BaseModel):
    summary: str


class Cool(BaseTool[CoolInput, CoolOutput]):
    name: ClassVar[str] = "Cool"
    description: ClassVar[str] = "Does the cool thing on a file. Always concurrency-safe."
    input_model: ClassVar[type[BaseModel]] = CoolInput
    max_result_size_chars: ClassVar[int] = 5_000

    def is_concurrency_safe(self, parsed: CoolInput) -> bool:
        return True

    def is_read_only(self, parsed: CoolInput) -> bool:
        return True

    async def call(self, parsed: CoolInput, ctx: ToolUseContext) -> ToolResult[CoolOutput]:
        body = (ctx.workdir / parsed.path).read_text()
        out = CoolOutput(summary=f"first 20 chars: {body[:20]!r}")
        return ToolResult(data=out, text=out.summary)
```

Then register it from `bootstrap.py` (or a plugin entry-point):

```python
from repo2rocm.tools.base import register_tool
from repo2rocm.tools.myteam.cool_tool import Cool
register_tool(Cool)
```

## Built-in tools

| Tool | Purpose | Concurrency-safe? |
|---|---|---|
| `Read` | Read a file with line numbers | always |
| `Grep` | ripgrep wrapper with Python fallback | always |
| `Glob` | List files by glob | always |
| `Edit` | Search-and-replace with staleness check | never |
| `Write` | Overwrite/create a file | never |
| `ApplyDiff` | Apply SEARCH/REPLACE hunks | never |
| `DockerExec` | Run a command inside the sandbox | input-dependent (read-only commands are safe) |
| `DockerCommit` | Snapshot the container | never |
| `DockerRollback` | Roll back to prior commit | never |
| `ChangeBaseImage` | Restart with a different base image | never |
| `ChangePythonVersion` | Restart with a different python | never |
| `WaitingListAdd` | Queue a package | never (writes shared queue) |
| `WaitingListAddFile` | Queue from a requirements file | never |
| `WaitingListShow` | Show queue | always |
| `WaitingListClear` | Clear queue | never |
| `ConflictListShow` | Show conflicts | always |
| `ConflictListSolve` | Resolve a conflict | never |
| `ConflictListClear` | Clear conflicts | never |
| `Download` | Batch-install pip + apt | never |
| `PyPIVersions` | Query PyPI | always |
| `DockerHubTags` | Query Docker Hub | always |
| `WebSearch` | Search the web (DuckDuckGo by default) | always |
| `Fetch` | HTTP GET | always |
| `EnvVerify` | Typed ROCm/CUDA verdict | never |
| `PaperVerify` | Typed paper-result verdict | always |
| `Agent` | Spawn a sub-agent | never |
| `SendMessage` | Talk to a running sub-agent | never |
| `TaskStop` | Kill a running sub-agent | never |

## Permission semantics

Every tool call goes through `core/permissions.py::resolve_permission`. The chain is:

1. Hook-supplied decision (PreToolUse hook returned allow/deny → final).
2. Rule matching: `always_deny > always_ask > always_allow`.
3. Tool-specific `check_permissions`.
4. Mode default — `PLAN` denies writes; `ACCEPT_EDITS` allows edits and reads; `BYPASS` allows all.
5. Interactive prompt (when applicable).

## Concurrency model

The `StreamingToolExecutor`:

* admits a tool to run iff `no_tools_running OR (new_is_safe AND all_running_are_safe)`
* yields results in **submission order** (the model sees results in the order it emitted the corresponding tool_use blocks)
* cancels sibling tools only on `DockerExec` / `Bash` errors
