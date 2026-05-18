# Permissions

Repo2ROCm v2 inherits Claude Code's permission-mode design (Ch. 1 & 6). The chain is
**single-resolution** — every tool call goes through the same function.

## Modes

| Mode | Behavior |
|---|---|
| `plan`              | Read-only. Mutations denied. |
| `default`           | Tool-specific checks; prompt user for unrecognized ops |
| `acceptEdits`       | Auto-allow file edits + reads; ask for other mutations |
| `auto`              | LLM classifier (not enabled yet) |
| `bypassPermissions` | Allow everything without prompting |
| `bubble`            | Sub-agent escalates to parent |

In Repo2ROCm:

* **Coordinator** runs in `plan` — it cannot touch files; it can only spawn sub-agents.
* **Explore / Planner / Verifier** run in `plan` — they cannot write.
* **Migrator** runs in `acceptEdits` — edits + bash auto-approved.
* **PaperReproducer** runs in `acceptEdits`.

## Quality gates as hooks (not booleans)

The old Repo2ROCm encoded gates as `_dockerhub_tags_seen`, `_pypi_versions_seen`,
`_gpu_check_seen` booleans inlined in the agent loop. v2 expresses them as
**PreToolUse callback hooks** registered at bootstrap. They are:

| Hook | Blocks |
|---|---|
| `before_change_base_image`     | `ChangeBaseImage(image)` unless `DockerHubTags(image)` was called previously |
| `before_pip_install_cuda_wheel`| `DockerExec("pip install <flash-attn|nvidia-*|bitsandbytes|xformers>...")` unless `PyPIVersions` was called previously |
| `before_env_verified`          | `EnvVerify` unless a GPU check (`torch.cuda.is_available` or `rocm-smi`) succeeded earlier |

Hook config is **frozen at startup** — modifications to `~/.repo2rocm/settings.json`
after `bootstrap()` are ignored, preventing TOCTOU attacks from arbitrary repos.

## CLI override

```bash
repo2rocm migrate owner/repo --permission-mode plan          # dry-run
repo2rocm migrate owner/repo --permission-mode acceptEdits   # default
repo2rocm migrate owner/repo --permission-mode bypassPermissions   # CI
```
