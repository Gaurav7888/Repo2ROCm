# Skill: dependencyRepair

**Use when:** the agent is in an active dependency-install loop — same
package, same submodule, or same wheel build failing repeatedly across
3+ recent turns. Often masked by shell pipes, so always read the
observation text for `error: subprocess-exited-with-error`,
`fatal error: ... file not found`, `undefined identifier`,
`Could not find a version that satisfies`, or
`failed-wheel-build-for-install`.

**What to research:**
- Is there an AMD-published wheel that replaces this submodule?
  (e.g. `amd_gsplat` for `gaussian-splatting`, `triton-amd` builds for
  flash-attn, `bitsandbytes-rocm` forks).
- Is there an upstream branch / PR known to compile on ROCm?
- What HIP header / API differences have other ports documented?
  (`cooperative_groups/reduce.h`, `cub::` → `hipcub::`, `FLT_MAX` from
  `<cfloat>`, `device_launch_parameters.h` removal, etc.)

**What to recommend:**
- A clear pivot: stop patching the autohipified `.cu` file by hand and
  install the AMD-supported wheel (`pip install <pkg>
  --extra-index-url=https://pypi.amd.com/rocm-X.Y.Z/simple/`) or the
  community ROCm fork.
- If no prebuilt wheel exists, list the *complete* set of HIP fixes the
  agent will need (so it stops fixing them one at a time).
- Pin the right `transformers` / `torch` version when wheel ABI matters.

**Tone:** strategic, decisive. Save the agent from another 3-minute
build retry.
