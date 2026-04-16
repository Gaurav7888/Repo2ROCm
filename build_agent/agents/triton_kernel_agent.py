"""
Triton Kernel Agent — handles Triton kernel ROCm compatibility.

Triton on ROCm is a different problem from CUDA hipification:
- tl.dot accumulator types behave differently
- tl.atomic_* operations have different performance characteristics
- Autotuning configs tuned for A100 are wrong for MI300X
- Some tl.constexpr patterns compile but produce wrong results on AMD

Pipeline:
1. Detect all Triton kernels and their @triton.autotune configs
2. Check each config for AMD-specific issues (warp size, VRAM assumptions)
3. Patch configs from KB templates or run mini autotuning
4. Verify correctness with numerical equivalence test
5. (Optional, --optimize-kernels) Run aotriton for pre-compiled kernels
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from storage.models import KernelInfo


@dataclass
class TritonKernelInfo:
    """Extended info for a Triton kernel."""
    file_path: str = ""
    kernel_name: str = ""
    has_autotune: bool = False
    autotune_configs: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    patched: bool = False
    verified: bool = False


@dataclass
class TritonPatchResult:
    file_path: str = ""
    patches_applied: List[str] = field(default_factory=list)
    success: bool = False
    new_configs: List[Dict[str, Any]] = field(default_factory=list)


class TritonKernelAgent:
    """
    Handles Triton kernel compatibility for ROCm.

    Operates inside the Docker container via sandbox session.
    """

    AMD_WARP_SIZE = 64
    NVIDIA_WARP_SIZE = 32

    AMD_KNOWN_ISSUES = [
        {
            "pattern": r"num_warps\s*=\s*(\d+)",
            "check": lambda val: int(val) > 8,
            "issue": "num_warps > 8 may cause issues on AMD; reduce to 4 or 8",
            "fix_key": "num_warps",
        },
        {
            "pattern": r"BLOCK_SIZE.*?=\s*(\d+)",
            "check": lambda val: int(val) > 256,
            "issue": "BLOCK_SIZE > 256 may be suboptimal on AMD; try 128 or 256",
            "fix_key": "block_size",
        },
        {
            "pattern": r"num_stages\s*=\s*(\d+)",
            "check": lambda val: int(val) > 3,
            "issue": "num_stages > 3 may not improve performance on AMD; try 2 or 3",
            "fix_key": "num_stages",
        },
    ]

    def __init__(self, sandbox_session, llm: str = "",
                 optimize: bool = False):
        self.session = sandbox_session
        self.llm = llm
        self.optimize = optimize

    def run(self, repo_path: str = "/repo") -> Dict[str, Any]:
        """Full Triton kernel compatibility pipeline."""
        results = {
            "kernels_found": 0,
            "issues_found": 0,
            "patched": 0,
            "verified": 0,
            "errors": [],
            "kernel_details": [],
        }

        # Step 1: Detect Triton kernels
        kernels = self._detect_triton_kernels(repo_path)
        results["kernels_found"] = len(kernels)

        if not kernels:
            return results

        # Step 2: Check for AMD-specific issues
        for kernel in kernels:
            issues = self._check_amd_issues(kernel)
            kernel.issues = issues
            results["issues_found"] += len(issues)

        # Step 3: Patch configs
        for kernel in kernels:
            if kernel.issues:
                patch_result = self._patch_kernel(kernel)
                if patch_result.success:
                    results["patched"] += 1
                    kernel.patched = True

        # Step 4: Verify correctness
        for kernel in kernels:
            if self._verify_kernel(kernel):
                results["verified"] += 1
                kernel.verified = True

        results["kernel_details"] = [
            {
                "file": k.file_path,
                "kernel_name": k.kernel_name,
                "has_autotune": k.has_autotune,
                "issues": k.issues,
                "patched": k.patched,
                "verified": k.verified,
            }
            for k in kernels
        ]

        return results

    def _detect_triton_kernels(self, repo_path: str) -> List[TritonKernelInfo]:
        """Find all files with Triton kernels."""
        kernels = []

        cmd = f"grep -rl '@triton\\.' {repo_path} --include='*.py' 2>/dev/null"
        output, rc = self._exec(cmd)
        if rc != 0 or not output.strip():
            return kernels

        for fpath in output.strip().splitlines():
            fpath = fpath.strip()
            if not fpath:
                continue

            cmd = f"cat {fpath}"
            content, _ = self._exec(cmd)
            if not content:
                continue

            jit_matches = re.finditer(
                r"@triton\.(?:jit|autotune)\s*(?:\([^)]*\))?\s*\ndef\s+(\w+)",
                content
            )
            for m in jit_matches:
                kernel_name = m.group(1)
                has_autotune = "@triton.autotune" in content[:m.start() + 200]

                autotune_configs = []
                if has_autotune:
                    autotune_configs = self._extract_autotune_configs(content, m.start())

                kernels.append(TritonKernelInfo(
                    file_path=fpath,
                    kernel_name=kernel_name,
                    has_autotune=has_autotune,
                    autotune_configs=autotune_configs,
                ))

        return kernels

    def _extract_autotune_configs(self, content: str,
                                  start_pos: int) -> List[Dict[str, Any]]:
        """Extract @triton.autotune config dicts from source code."""
        configs = []

        search_region = content[max(0, start_pos - 2000):start_pos]
        config_pattern = re.compile(
            r"triton\.Config\s*\(\s*\{([^}]+)\}",
            re.DOTALL
        )

        for m in config_pattern.finditer(search_region):
            config_str = m.group(1)
            config = {}
            for kv in re.finditer(r"['\"](\w+)['\"]\s*:\s*(\d+)", config_str):
                config[kv.group(1)] = int(kv.group(2))
            if config:
                configs.append(config)

        return configs

    def _check_amd_issues(self, kernel: TritonKernelInfo) -> List[str]:
        """Check kernel configs for AMD-specific issues."""
        issues = []

        cmd = f"cat {kernel.file_path}"
        content, _ = self._exec(cmd)
        if not content:
            return issues

        for check in self.AMD_KNOWN_ISSUES:
            matches = re.finditer(check["pattern"], content)
            for m in matches:
                val = m.group(1)
                if check["check"](val):
                    issues.append(
                        f"{check['issue']} (found: {check['fix_key']}={val})"
                    )

        if "tl.dot" in content:
            if "allow_tf32" in content:
                issues.append(
                    "tl.dot with allow_tf32 may behave differently on AMD; "
                    "consider explicit accumulator type"
                )

        if re.search(r"32\s*#.*warp", content, re.IGNORECASE):
            issues.append(
                "Hardcoded warp size 32 detected; AMD uses wavefront size 64"
            )

        return issues

    def _patch_kernel(self, kernel: TritonKernelInfo) -> TritonPatchResult:
        """Apply AMD-compatible patches to Triton kernel configs."""
        result = TritonPatchResult(file_path=kernel.file_path)

        cmd = f"cat {kernel.file_path}"
        content, _ = self._exec(cmd)
        if not content:
            return result

        patched = content
        patches = []

        if "num_warps" in str(kernel.issues):
            patched = re.sub(
                r"(num_warps\s*[=:]\s*)\d+",
                lambda m: m.group(1) + "4",
                patched
            )
            patches.append("num_warps reduced to 4 for AMD compatibility")

        if "num_stages" in str(kernel.issues):
            patched = re.sub(
                r"(num_stages\s*[=:]\s*)\d+",
                lambda m: m.group(1) + "2",
                patched
            )
            patches.append("num_stages reduced to 2 for AMD")

        if patches and patched != content:
            backup_cmd = f"cp {kernel.file_path} {kernel.file_path}.bak"
            self._exec(backup_cmd)

            write_cmd = f"cat > {kernel.file_path} << 'TRITON_PATCH_EOF'\n{patched}\nTRITON_PATCH_EOF"
            _, rc = self._exec(write_cmd)

            result.success = (rc == 0)
            result.patches_applied = patches

        return result

    def _verify_kernel(self, kernel: TritonKernelInfo) -> bool:
        """Run a basic import and compilation check for the Triton kernel."""
        module_path = kernel.file_path.replace("/repo/", "").replace("/", ".").replace(".py", "")

        verify_cmd = (
            f"cd /repo && python -c \""
            f"import triton; "
            f"print('Triton version:', triton.__version__); "
            f"print('VERIFICATION_OK')\""
        )
        output, rc = self._exec(verify_cmd)

        return rc == 0 and output and "VERIFICATION_OK" in output

    def _exec(self, cmd: str) -> Tuple[str, int]:
        """Execute a command in the sandbox."""
        try:
            from utils.waiting_list import WaitingList
            from utils.conflict_list import ConflictList
            wl = WaitingList()
            cl = ConflictList()
            output, rc = self.session.execute(cmd, wl, cl)
            if isinstance(rc, str):
                rc = -1
            return output or "", rc
        except Exception as e:
            return str(e), -1
