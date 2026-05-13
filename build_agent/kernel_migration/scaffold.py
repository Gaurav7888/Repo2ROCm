"""
Correctness-first CUDA kernel migration scaffold.

This is intentionally not an optimizer. It lays out the research lane for a
future sub-agent:

1. Discover CUDA/CUDA-extension sources.
2. Run hipify in examine mode to scope the migration.
3. Apply hipify with conservative output/in-place commands.
4. Run a granular fixer pass for common post-hipify correctness issues.
5. Compile/import-check only; performance tuning is out of scope.

The module is designed to be usable in dry-run mode today and easy to wire into
the sandbox later by passing an executor callable.
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple


Executor = Callable[[str], "CommandResult"]

_CUDA_EXTS = {".cu", ".cuh", ".h", ".hpp", ".cc", ".cpp"}
_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist",
    "site-packages", "graphify-out", "checkpoints", "wandb", "outputs",
}


class MigrationStage(str, Enum):
    DISCOVER = "discover"
    HIPIFY_EXAMINE = "hipify_examine"
    HIPIFY_APPLY = "hipify_apply"
    GRANULAR_FIX = "granular_fix"
    COMPILE_CHECK = "compile_check"
    REPORT = "report"


@dataclass
class CommandResult:
    command: str
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.return_code == 0


@dataclass
class KernelCandidate:
    path: str
    kind: str = "cuda"
    purpose: str = "other"
    includes: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FixSuggestion:
    file_path: str
    issue: str
    rationale: str
    patch_hint: str
    confidence: float = 0.5
    requires_subagent: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class KernelMigrationReport:
    repo_path: str
    dry_run: bool = True
    stages_completed: List[str] = field(default_factory=list)
    candidates: List[KernelCandidate] = field(default_factory=list)
    commands_planned: List[str] = field(default_factory=list)
    command_results: List[CommandResult] = field(default_factory=list)
    fix_suggestions: List[FixSuggestion] = field(default_factory=list)
    compile_commands: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "repo_path": self.repo_path,
            "dry_run": self.dry_run,
            "stages_completed": list(self.stages_completed),
            "candidates": [c.to_dict() for c in self.candidates],
            "commands_planned": list(self.commands_planned),
            "command_results": [asdict(r) for r in self.command_results],
            "fix_suggestions": [f.to_dict() for f in self.fix_suggestions],
            "compile_commands": list(self.compile_commands),
            "errors": list(self.errors),
        }


def _read_text(path: Path, max_chars: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except OSError:
        return ""


def _iter_candidate_files(repo_path: Path) -> Iterable[Path]:
    for root_str, dirs, files in os.walk(repo_path):
        root = Path(root_str)
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in files:
            path = root / name
            if path.suffix.lower() in _CUDA_EXTS:
                yield path


def _looks_cuda_related(path: Path, text: str) -> bool:
    if path.suffix.lower() in {".cu", ".cuh"}:
        return True
    markers = (
        "__global__", "__device__", "__host__", "cuda_runtime.h",
        "cuda_fp16.h", "cudaMalloc", "cudaMemcpy", "cudaLaunchKernel",
        "threadIdx", "blockIdx", "blockDim", "gridDim",
    )
    return any(marker in text for marker in markers)


def _classify_purpose(path: Path, text: str) -> str:
    hay = f"{path.name}\n{text[:8000]}".lower()
    checks = [
        ("attention", ("attention", "flash_attn", "scaled_dot")),
        ("normalization", ("layernorm", "rmsnorm", "batchnorm", "norm")),
        ("positional_encoding", ("rotary", "rope", "position")),
        ("optimizer", ("adam", "optimizer", "momentum")),
        ("quantization", ("quant", "int8", "int4", "dequant")),
        ("sampling", ("sampling", "topk", "softmax")),
        ("fused_operation", ("fused", "fusion")),
    ]
    for purpose, markers in checks:
        if any(marker in hay for marker in markers):
            return purpose
    return "other"


def _extract_includes(text: str) -> List[str]:
    out: List[str] = []
    for match in re.finditer(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", text, re.MULTILINE):
        out.append(match.group(1))
    return out[:40]


def _extract_symbols(text: str) -> List[str]:
    symbols = set()
    for pattern in (
        r"\b__global__\s+[\w:<>,\s*&]+\s+(\w+)\s*\(",
        r"\b__device__\s+[\w:<>,\s*&]+\s+(\w+)\s*\(",
        r"\bAT_DISPATCH_[A-Z0-9_]+\s*\(",
    ):
        for match in re.finditer(pattern, text):
            symbols.add(match.group(1) if match.lastindex else match.group(0))
    return sorted(symbols)[:40]


def _risk_flags(text: str) -> List[str]:
    flags = []
    checks = [
        ("inline_ptx", r"\basm\s*(?:volatile)?\s*\("),
        ("texture_surface_api", r"\b(texture|surface)<|cudaTextureObject_t|cudaSurfaceObject_t"),
        ("warp_size_assumption", r"\b(warpSize|WARP_SIZE|warp_size)\b|[\s(]32\s*[),;]\s*//.*warp"),
        ("cuda_arch_guard", r"__CUDA_ARCH__|CUDA_VERSION"),
        ("nvidia_library_header", r"cublas_v2\.h|cusparse\.h|curand\.h|nccl\.h"),
        ("driver_api", r"\bcu(Module|Ctx|Launch|Memcpy|MemAlloc|Event|Stream)"),
        ("cooperative_groups", r"cooperative_groups"),
    ]
    for name, pattern in checks:
        if re.search(pattern, text, flags=re.IGNORECASE):
            flags.append(name)
    return flags


def discover_cuda_sources(repo_path: str) -> List[KernelCandidate]:
    """Discover CUDA-related files without modifying the repository."""
    root = Path(repo_path).resolve()
    candidates: List[KernelCandidate] = []
    if not root.exists():
        return candidates

    for path in _iter_candidate_files(root):
        text = _read_text(path)
        if not _looks_cuda_related(path, text):
            continue
        rel = str(path.relative_to(root))
        candidates.append(KernelCandidate(
            path=rel,
            kind="cuda",
            purpose=_classify_purpose(path, text),
            includes=_extract_includes(text),
            symbols=_extract_symbols(text),
            risk_flags=_risk_flags(text),
        ))
    return candidates


class HipifyCommandBuilder:
    """Build conservative hipify commands for a candidate file."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    def absolute_path(self, candidate: KernelCandidate) -> str:
        return str(self.repo_path / candidate.path)

    def examine(self, candidate: KernelCandidate) -> str:
        path = self.absolute_path(candidate)
        return (
            f"hipify-clang --examine {path} 2>&1 "
            f"|| hipify-perl --examine {path} 2>&1"
        )

    def apply(self, candidate: KernelCandidate, inplace: bool = False) -> str:
        path = self.absolute_path(candidate)
        if inplace:
            return (
                f"cp {path} {path}.prehip && "
                f"(hipify-clang --inplace {path} 2>&1 "
                f"|| hipify-perl --inplace {path} 2>&1)"
            )
        out_path = _hip_output_path(path)
        return (
            f"hipify-clang {path} -o {out_path} 2>&1 "
            f"|| hipify-perl {path} > {out_path} 2>&1"
        )


def _hip_output_path(path: str) -> str:
    p = Path(path)
    if p.suffix == ".cu":
        return str(p.with_suffix(".hip.cpp"))
    if p.suffix == ".cuh":
        return str(p.with_suffix(".hip.h"))
    return str(p) + ".hip"


class GranularIssueFixer:
    """
    Rule-based post-hipify fixer planner.

    The future sub-agent should consume these suggestions, inspect the specific
    file context, and apply minimal correctness patches. The current scaffold
    deliberately returns patch hints rather than rewriting code blindly.
    """

    def suggest(self, candidate: KernelCandidate, source_text: str,
                hipify_output: str = "") -> List[FixSuggestion]:
        suggestions: List[FixSuggestion] = []
        path = candidate.path
        hay = source_text + "\n" + (hipify_output or "")

        if re.search(r"cuda_runtime\.h", hay):
            suggestions.append(FixSuggestion(
                file_path=path,
                issue="cuda_runtime_header",
                rationale="hipify usually maps runtime APIs, but includes often need explicit HIP runtime headers.",
                patch_hint='Replace `#include <cuda_runtime.h>` with `#include <hip/hip_runtime.h>`.',
                confidence=0.9,
            ))

        if re.search(r"cuda_fp16\.h", hay):
            suggestions.append(FixSuggestion(
                file_path=path,
                issue="cuda_fp16_header",
                rationale="Half-precision CUDA header should use HIP's fp16 header on ROCm.",
                patch_hint='Replace `#include <cuda_fp16.h>` with `#include <hip/hip_fp16.h>`.',
                confidence=0.85,
            ))

        if re.search(r"\b(cublas_v2|cusparse|curand|nccl)\.h", hay):
            suggestions.append(FixSuggestion(
                file_path=path,
                issue="nvidia_library_header",
                rationale="CUDA library headers need library-level ROCm equivalents and usually API call review.",
                patch_hint="Map cuBLAS/cuSPARSE/cuRAND/NCCL headers and calls to hipBLAS/rocSPARSE/rocRAND/RCCL equivalents.",
                confidence=0.65,
                requires_subagent=True,
            ))

        if "inline_ptx" in candidate.risk_flags:
            suggestions.append(FixSuggestion(
                file_path=path,
                issue="inline_ptx",
                rationale="Inline PTX does not translate mechanically to AMDGCN and must be rewritten or replaced.",
                patch_hint="Isolate the asm block, replace with HIP/C++ intrinsics or mark unsupported with a guarded fallback.",
                confidence=0.95,
                requires_subagent=True,
            ))

        if "warp_size_assumption" in candidate.risk_flags:
            suggestions.append(FixSuggestion(
                file_path=path,
                issue="warp_size_assumption",
                rationale="AMD wavefronts are commonly 64 lanes; CUDA kernels often bake in 32-lane warp assumptions.",
                patch_hint="Replace hardcoded warp-size constants with `warpSize` or a HIP runtime/device conditional; verify reductions and shuffles.",
                confidence=0.8,
                requires_subagent=True,
            ))

        if re.search(r"__shfl(_sync)?|__ballot(_sync)?|__activemask", hay):
            suggestions.append(FixSuggestion(
                file_path=path,
                issue="warp_intrinsics",
                rationale="HIP has shuffle/vote equivalents, but masks and wavefront width can change correctness.",
                patch_hint="Review shuffle/vote intrinsics after hipify; add tests for boundary lanes and non-32 multiples.",
                confidence=0.75,
                requires_subagent=True,
            ))

        if "texture_surface_api" in candidate.risk_flags:
            suggestions.append(FixSuggestion(
                file_path=path,
                issue="texture_surface_api",
                rationale="CUDA texture/surface APIs are a common hipify gap and need manual resource binding review.",
                patch_hint="Replace texture/surface object usage with supported HIP texture APIs or explicit memory loads.",
                confidence=0.7,
                requires_subagent=True,
            ))

        if re.search(r"__CUDA_ARCH__|CUDA_VERSION", hay):
            suggestions.append(FixSuggestion(
                file_path=path,
                issue="cuda_preprocessor_guards",
                rationale="CUDA-specific guards may exclude HIP code paths or select NVIDIA-only implementations.",
                patch_hint="Add `__HIP_PLATFORM_AMD__` / `__HIPCC__` branches and keep CUDA branches intact for portability.",
                confidence=0.8,
                requires_subagent=True,
            ))

        if "hipify" in hipify_output.lower() and "warning" in hipify_output.lower():
            suggestions.append(FixSuggestion(
                file_path=path,
                issue="hipify_warning_review",
                rationale="hipify warnings indicate APIs or macros that were not confidently converted.",
                patch_hint="Have the kernel-migration sub-agent inspect each hipify warning and apply minimal correctness fixes.",
                confidence=0.6,
                requires_subagent=True,
            ))

        return suggestions


class CompileCommandBuilder:
    """Prepare correctness-only compile/import checks."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    def compile_object(self, candidate: KernelCandidate) -> str:
        source = self.repo_path / candidate.path
        hip_source = _hip_output_path(str(source))
        obj = str(Path(hip_source).with_suffix(".o"))
        return f"hipcc -c {hip_source} -o {obj} -I/opt/rocm/include 2>&1"

    def python_extension_probe(self) -> str:
        return (
            "python - <<'PY'\n"
            "import torch\n"
            "print('torch', torch.__version__, 'hip', getattr(torch.version, 'hip', None))\n"
            "assert torch.cuda.is_available(), 'ROCm GPU is not visible to PyTorch'\n"
            "print('KERNEL_MIGRATION_IMPORT_PROBE_OK')\n"
            "PY"
        )


class KernelMigrationAgent:
    """
    Sub-agent scaffold for correctness-only CUDA-to-HIP migration.

    Pass `dry_run=True` to produce an auditable plan without changing files.
    Pass an executor callable later to run inside the sandbox:

        def executor(cmd: str) -> CommandResult: ...
        KernelMigrationAgent('/repo', executor=executor, dry_run=False).run()
    """

    def __init__(self, repo_path: str = "/repo",
                 executor: Optional[Executor] = None,
                 dry_run: bool = True,
                 inplace: bool = False):
        self.repo_path = str(Path(repo_path).resolve())
        self.executor = executor
        self.dry_run = dry_run
        self.inplace = inplace
        self.hipify = HipifyCommandBuilder(self.repo_path)
        self.fixer = GranularIssueFixer()
        self.compiler = CompileCommandBuilder(self.repo_path)

    def run(self) -> KernelMigrationReport:
        report = KernelMigrationReport(repo_path=self.repo_path, dry_run=self.dry_run)

        candidates = discover_cuda_sources(self.repo_path)
        report.candidates = candidates
        report.stages_completed.append(MigrationStage.DISCOVER.value)
        if not candidates:
            return report

        for candidate in candidates:
            examine_cmd = self.hipify.examine(candidate)
            apply_cmd = self.hipify.apply(candidate, inplace=self.inplace)
            report.commands_planned.extend([examine_cmd, apply_cmd])

            hipify_output = ""
            if not self.dry_run and self.executor is not None:
                examine_result = self.executor(examine_cmd)
                report.command_results.append(examine_result)
                hipify_output += examine_result.stdout + "\n" + examine_result.stderr
                apply_result = self.executor(apply_cmd)
                report.command_results.append(apply_result)
                hipify_output += apply_result.stdout + "\n" + apply_result.stderr
                if not apply_result.ok:
                    report.errors.append(f"hipify failed for {candidate.path}")

            source_text = _read_text(Path(self.repo_path) / candidate.path)
            report.fix_suggestions.extend(
                self.fixer.suggest(candidate, source_text, hipify_output)
            )

            compile_cmd = self.compiler.compile_object(candidate)
            report.compile_commands.append(compile_cmd)
            if not self.dry_run and self.executor is not None:
                compile_result = self.executor(compile_cmd)
                report.command_results.append(compile_result)
                if not compile_result.ok:
                    report.errors.append(f"compile check failed for {candidate.path}")

        report.stages_completed.extend([
            MigrationStage.HIPIFY_EXAMINE.value,
            MigrationStage.HIPIFY_APPLY.value,
            MigrationStage.GRANULAR_FIX.value,
            MigrationStage.COMPILE_CHECK.value,
            MigrationStage.REPORT.value,
        ])
        return report
