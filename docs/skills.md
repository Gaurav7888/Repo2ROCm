# Skills

A skill is a Markdown file with YAML frontmatter. At startup, only the frontmatter is
loaded into the system prompt (a compact menu); the body is loaded on invocation via
`/<skill-name>`.

## Authoring a skill

Drop a directory containing `SKILL.md` into one of:

| Path | Source | Trust |
|---|---|---|
| `/etc/repo2rocm/skills/`             | policy   | enterprise |
| `~/.repo2rocm/skills/`               | user     | per-user |
| `./.repo2rocm/skills/`               | project  | committed to repo |
| `repo2rocm/skills/builtin/`          | builtin  | shipped |

Example `SKILL.md`:

```markdown
---
name: my_skill
description: One sentence the Coordinator sees.
when_to_use: Multi-line guidance for the model.
allowed_tools: [Read, Grep, DockerExec]
paths: ["packages/database/**"]   # optional conditional activation
hooks:
  PreToolUse:
    - matcher: Bash
      hooks:
        - { type: command, command: "${REPO2ROCM_SKILL_DIR}/validate.sh", once: true }
---

# My Skill

Markdown body. Loaded only when `/my_skill` is invoked.
```

## Built-in skills shipped

| Name | Purpose |
|---|---|
| `rocm_image_catalog`     | Authoritative catalog of ROCm Docker images |
| `cuda_to_rocm_mapping`   | CUDA-only PyPI wheel → AMD equivalent table |
| `banned_nvidia_packages` | Wheels that NEVER work on AMD |
| `flash_attn_amd_install` | How to install flash-attention on AMD ROCm |
| `py312_compat`           | Python 3.12 breakage patterns |
| `hipify_patterns`        | `.cu` / `.cpp` patterns hipify can't auto-translate |

## Conditional activation (paths)

If a skill declares `paths: ["packages/database/**"]`, it activates only when an agent's
tool calls touch a path matching the glob. This avoids polluting the menu for unrelated
tasks.

## Skill-declared hooks

Skill frontmatter may declare hooks. They are registered as **session-scoped** hooks when
the skill is invoked. `Stop` hooks on subagents are automatically converted to
`SubagentStop`.
