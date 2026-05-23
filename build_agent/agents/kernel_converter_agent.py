"""
Correctness-only CUDA-to-HIP kernel converter specialist.

This is **not** the older ``cuda_kernel_agent.CUDAKernelAgent`` and does not
replace it. ``cuda_kernel_agent.py`` keeps its existing inventory + hipify +
optional optimization pipeline; ``KernelConverterAgent`` is the new lane
described in ``docs/next_research_plan.md`` (Track 2): a granular,
correctness-only repairer that runs through the scaffold (``hipify examine``
→ ``hipify apply`` → ``granular fix`` → ``hipcc -c``) and emits a structured
``storage.models.KernelMigrationReport``.

Forbidden in this agent (research plan):
    - performance tuning / autotuning
    - large rewrites of the original kernel
    - mock-success / fabricated output

The agent is idempotent: callers should construct it once per build attempt.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from kernel_migration.scaffold import (
    CommandResult,
    CompileCommandBuilder,
    Executor,
    FixSuggestion,
    GranularIssueFixer,
    HipifyCommandBuilder,
    KernelCandidate,
    _hip_output_path,
    _read_text,
    discover_cuda_sources,
)
from storage.models import KernelMigrationReport


# ── Constants from the research plan ─────────────────────────────────────────

ALLOWED_SCOPE = "correctness_only"
FORBIDDEN_ACTIONS: List[str] = [
    "performance tuning",
    "large rewrites",
    "mock success",
]

# Map fix issue → (search, replace) for the rule-based, non-subagent path.
_RULE_BASED_PATCHES: Dict[str, List[Dict[str, str]]] = {
    "cuda_runtime_header": [
        {"search": "#include <cuda_runtime.h>", "replace": "#include <hip/hip_runtime.h>"},
        {"search": '#include "cuda_runtime.h"', "replace": '#include "hip/hip_runtime.h"'},
    ],
    "cuda_fp16_header": [
        {"search": "#include <cuda_fp16.h>", "replace": "#include <hip/hip_fp16.h>"},
        {"search": '#include "cuda_fp16.h"', "replace": '#include "hip/hip_fp16.h"'},
    ],
}


def _looks_like_pytorch_extension(repo_root: Path) -> bool:
    """Cheap heuristic: setup.py mentions CUDAExtension / cpp_extension."""
    setup = repo_root / "setup.py"
    if not setup.exists():
        return False
    try:
        text = setup.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    needles = ("CUDAExtension", "cpp_extension", "BuildExtension")
    return any(n in text for n in needles)


def build_subagent_task_packet(
    candidate: KernelCandidate,
    suggestion: FixSuggestion,
    hipify_output: str,
    compile_error: str = "",
) -> Dict[str, Any]:
    """Return the structured task packet from the research plan.

    Shape (verbatim from ``docs/next_research_plan.md``)::

        {
          "candidate": "src/kernels/attention.cu",
          "hipify_output": "...",
          "compile_error": "...",
          "risk_flags": ["inline_ptx", "warp_size_assumption"],
          "allowed_scope": "correctness_only",
          "forbidden": ["performance tuning", "large rewrites", "mock success"]
        }

    Plus an ``issue`` and ``rationale`` field carried over from the
    scaffold's ``FixSuggestion`` so an LLM can ground its edit.
    """
    return {
        "candidate": candidate.path,
        "issue": suggestion.issue,
        "rationale": suggestion.rationale,
        "patch_hint": suggestion.patch_hint,
        "hipify_output": hipify_output[-4000:] if hipify_output else "",
        "compile_error": compile_error[-2000:] if compile_error else "",
        "risk_flags": list(candidate.risk_flags),
        "allowed_scope": ALLOWED_SCOPE,
        "forbidden": list(FORBIDDEN_ACTIONS),
    }


class KernelConverterAgent:
    """Correctness-only CUDA-to-HIP migration specialist.

    Parameters
    ----------
    executor:
        Sandbox executor matching the scaffold's
        ``Callable[[str], CommandResult]`` interface.
    repo_root:
        Absolute path to the repository to migrate. The agent never goes
        outside this directory.
    llm:
        Optional ``Callable[[Dict[str, Any]], Dict[str, Any]]`` invoked for
        ``FixSuggestion.requires_subagent=True`` items. Must return a JSON
        shape with at least ``{"patches": [{"file", "search", "replace"}, ...]}``.
        When omitted, ``requires_subagent`` items are emitted into the
        report's manual-fix list instead.
    dry_run:
        If True, no commands are executed; the executor is only used to
        record planned commands. ``hipcc`` is never invoked.
    report_dir:
        Where to write ``kernel_migration_report.json``. Created if missing.
    """

    REPORT_BASENAME = "kernel_migration_report.json"

    def __init__(
        self,
        executor: Executor,
        repo_root: str,
        llm: Optional[Callable[[Dict[str, Any]], Any]] = None,
        dry_run: bool = False,
        report_dir: Optional[str] = None,
        repo_id: str = "",
        attempt_id: str = "",
        container_repo_root: Optional[str] = None,
    ):
        self.executor = executor
        self.repo_root = str(Path(repo_root).resolve())
        # Repo path INSIDE the sandbox/container. The scaffold builders ship
        # absolute paths into hipify/hipcc commands; when we run inside Docker
        # the host path is invisible, so we redirect the command-builder side
        # to the container path while keeping the host path for filesystem
        # walks, code reads, and edit-block patches.
        self.container_repo_root = (
            str(Path(container_repo_root)) if container_repo_root else self.repo_root
        )
        self.llm = llm
        self.dry_run = bool(dry_run)
        self.report_dir = str(Path(report_dir).resolve()) if report_dir else self.repo_root
        self.repo_id = repo_id
        self.attempt_id = attempt_id

        # Command builders point at the container path so commands sent
        # through ``executor`` reach files inside the container.
        self.hipify = HipifyCommandBuilder(self.container_repo_root)
        self.fixer = GranularIssueFixer()
        self.compiler = CompileCommandBuilder(self.container_repo_root)

    # ── Toolchain detection ─────────────────────────────────────────────────

    def _toolchain_available(self) -> Dict[str, bool]:
        """Return availability flags for ``hipify-clang``, ``hipify-perl``, ``hipcc``."""
        availability = {"hipify_clang": False, "hipify_perl": False, "hipcc": False}
        if self.dry_run:
            # In dry-run we can't actually probe; assume nothing is present so
            # the report path treats this as "planned only". Tests can override
            # via a custom executor.
            return availability
        probes = {
            "hipify_clang": "command -v hipify-clang >/dev/null 2>&1 && echo OK",
            "hipify_perl":  "command -v hipify-perl >/dev/null 2>&1 && echo OK",
            "hipcc":        "command -v hipcc >/dev/null 2>&1 && echo OK",
        }
        for key, cmd in probes.items():
            try:
                res = self.executor(cmd)
            except Exception:
                continue
            if res and res.return_code == 0 and "OK" in (res.stdout or ""):
                availability[key] = True
        return availability

    # ── Phase implementations ───────────────────────────────────────────────

    def _phase_inventory(
        self, report: KernelMigrationReport
    ) -> List[KernelCandidate]:
        candidates = discover_cuda_sources(self.repo_root)
        report.n_kernels = len(candidates)
        unique_flags = sorted({f for c in candidates for f in c.risk_flags})
        report.risk_flags = unique_flags
        report.evidence.append(
            f"inventory: discovered {len(candidates)} CUDA source(s) under {self.repo_root}"
        )
        return candidates

    def _phase_examine(
        self,
        report: KernelMigrationReport,
        candidates: List[KernelCandidate],
    ) -> Dict[str, str]:
        """Run hipify --examine per file. Returns map ``rel_path → output``."""
        per_file_output: Dict[str, str] = {}
        for candidate in candidates:
            cmd = self.hipify.examine(candidate)
            if self.dry_run:
                self.executor(cmd)
                per_file_output[candidate.path] = ""
                continue
            try:
                result = self.executor(cmd)
            except Exception as exc:
                report.errors.append(f"examine error for {candidate.path}: {exc}")
                per_file_output[candidate.path] = ""
                continue
            output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
            per_file_output[candidate.path] = output
            report.kernels_examined += 1
            if not result.ok:
                report.errors.append(
                    f"hipify --examine non-zero rc={result.return_code} for {candidate.path}"
                )
            unsupported = [
                line.strip()
                for line in output.splitlines()
                if "warning" in line.lower() or "unsupported" in line.lower()
            ][:20]
            if unsupported:
                report.evidence.append(
                    f"examine[{candidate.path}]: {len(unsupported)} warning/unsupported lines"
                )
        return per_file_output

    def _phase_apply(
        self,
        report: KernelMigrationReport,
        candidates: List[KernelCandidate],
        examine_output: Dict[str, str],
    ) -> Dict[str, str]:
        """Run hipify (non-in-place) per file. Returns map ``rel_path → output``."""
        per_file_apply: Dict[str, str] = {}
        for candidate in candidates:
            inplace = self._compile_strategy_clear(candidate, examine_output.get(candidate.path, ""))
            cmd = self.hipify.apply(candidate, inplace=inplace)
            if self.dry_run:
                self.executor(cmd)
                per_file_apply[candidate.path] = ""
                continue
            try:
                result = self.executor(cmd)
            except Exception as exc:
                report.errors.append(f"apply error for {candidate.path}: {exc}")
                per_file_apply[candidate.path] = ""
                continue
            per_file_apply[candidate.path] = (result.stdout or "") + (
                "\n" + result.stderr if result.stderr else ""
            )
            if result.ok:
                report.kernels_applied += 1
            else:
                report.errors.append(
                    f"hipify apply non-zero rc={result.return_code} for {candidate.path}"
                )
        return per_file_apply

    def _compile_strategy_clear(
        self, candidate: KernelCandidate, examine_output: str
    ) -> bool:
        """Decide between in-place and side-by-side hipify.

        The research plan says: prefer non-in-place first; only fall through
        to in-place when the compile strategy is clear. Today "clear" means
        the candidate has no high-risk flags AND examine produced no
        warnings — which is conservative on purpose.
        """
        if candidate.risk_flags:
            return False
        if "warning" in (examine_output or "").lower():
            return False
        # Even when clear we keep non-in-place so the original file stays
        # available for diff comparison. Returning False is the safe default.
        return False

    def _phase_granular_fix(
        self,
        report: KernelMigrationReport,
        candidates: List[KernelCandidate],
        examine_output: Dict[str, str],
        apply_output: Dict[str, str],
    ) -> None:
        """Apply rule-based correctness patches; emit subagent task packets."""
        for candidate in candidates:
            source_path = Path(self.repo_root) / candidate.path
            source_text = _read_text(source_path)
            hip_path = Path(_hip_output_path(str(source_path)))
            hip_text = _read_text(hip_path) if hip_path.exists() else ""
            hipify_blob = (examine_output.get(candidate.path, "") + "\n"
                           + apply_output.get(candidate.path, ""))
            suggestions = self.fixer.suggest(candidate, source_text, hipify_blob)

            for suggestion in suggestions:
                if not suggestion.requires_subagent:
                    applied = self._apply_rule_based_patch(suggestion, hip_path, hip_text)
                    if applied is not None:
                        report.granular_fixes_applied.append(applied)
                        # Refresh hip_text after a successful patch.
                        if applied.get("status") == "applied" and hip_path.exists():
                            hip_text = _read_text(hip_path)
                    continue

                packet = build_subagent_task_packet(
                    candidate, suggestion, hipify_blob, compile_error=""
                )
                if self.llm is not None:
                    llm_record = self._invoke_llm(packet)
                    if llm_record is not None:
                        applied_count = self._apply_llm_patches(
                            llm_record.get("patches", []) or [], report
                        )
                        report.granular_fixes_applied.append({
                            "issue": suggestion.issue,
                            "file": candidate.path,
                            "status": "llm_applied" if applied_count > 0 else "llm_no_patch",
                            "applied_count": applied_count,
                            "task_packet": packet,
                        })
                        # Refresh hip_text after potential LLM edits.
                        hip_text = _read_text(hip_path) if hip_path.exists() else hip_text
                        continue

                report.manual_fix_count += 1
                report.granular_fixes_applied.append({
                    "issue": suggestion.issue,
                    "file": candidate.path,
                    "status": "manual_required",
                    "rationale": suggestion.rationale,
                    "patch_hint": suggestion.patch_hint,
                    "task_packet": packet,
                })

    def _apply_rule_based_patch(
        self,
        suggestion: FixSuggestion,
        hip_path: Path,
        hip_text: str,
    ) -> Optional[Dict[str, Any]]:
        """Apply one of the deterministic header rewrites via tools.code_edit."""
        rules = _RULE_BASED_PATCHES.get(suggestion.issue)
        if not rules or not hip_path.exists() or not hip_text:
            return {
                "issue": suggestion.issue,
                "file": suggestion.file_path,
                "status": "skipped_no_target",
                "rationale": suggestion.rationale,
            }
        # Lazy import: tools.code_edit pulls in optional modules.
        try:
            from tools.code_edit import apply_edit  # type: ignore
        except Exception as exc:
            return {
                "issue": suggestion.issue,
                "file": suggestion.file_path,
                "status": "skipped_no_editor",
                "error": str(exc),
            }

        for rule in rules:
            if rule["search"] not in hip_text:
                continue
            result = apply_edit(str(hip_path), hip_text, rule["search"], rule["replace"])
            if isinstance(result, dict) and result.get("message") == "succeed":
                return {
                    "issue": suggestion.issue,
                    "file": suggestion.file_path,
                    "status": "applied",
                    "search": rule["search"],
                    "replace": rule["replace"],
                }
        return {
            "issue": suggestion.issue,
            "file": suggestion.file_path,
            "status": "no_match",
            "rationale": suggestion.rationale,
        }

    def _invoke_llm(self, packet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call the LLM with the structured task packet. Tolerant of failures."""
        if self.llm is None:
            return None
        try:
            response = self.llm(packet)
        except Exception as exc:
            return {"error": f"llm_invocation_failed: {exc}", "patches": []}
        if response is None:
            return None
        if isinstance(response, dict):
            return response
        if isinstance(response, str):
            try:
                return json.loads(response)
            except Exception:
                return {"raw": response, "patches": []}
        return {"patches": []}

    def _apply_llm_patches(
        self, patches: List[Dict[str, Any]], report: KernelMigrationReport
    ) -> int:
        """Apply ``[{file, search, replace}]`` patches; return count succeeded."""
        try:
            from tools.code_edit import apply_edit  # type: ignore
        except Exception as exc:
            report.errors.append(f"code_edit unavailable: {exc}")
            return 0
        applied = 0
        for patch in patches:
            file_path = patch.get("file") or patch.get("file_path") or ""
            search = patch.get("search") or patch.get("before") or ""
            replace = patch.get("replace") or patch.get("after") or ""
            if not file_path or not search:
                continue
            target = Path(file_path)
            if not target.is_absolute():
                target = Path(self.repo_root) / file_path
            if not target.exists():
                continue
            # Safety: never edit outside repo_root.
            try:
                target.resolve().relative_to(Path(self.repo_root).resolve())
            except ValueError:
                report.errors.append(f"refused out-of-repo patch: {target}")
                continue
            try:
                content = target.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            result = apply_edit(str(target), content, search, replace)
            if isinstance(result, dict) and result.get("message") == "succeed":
                applied += 1
        return applied

    def _phase_verify(
        self,
        report: KernelMigrationReport,
        candidates: List[KernelCandidate],
    ) -> None:
        if self.dry_run:
            for candidate in candidates:
                cmd = self.compiler.compile_object(candidate)
                self.executor(cmd)
            return
        for candidate in candidates:
            cmd = self.compiler.compile_object(candidate)
            try:
                result = self.executor(cmd)
            except Exception as exc:
                report.errors.append(f"hipcc invocation error for {candidate.path}: {exc}")
                report.compile_failed += 1
                continue
            if result.ok:
                report.compile_passed += 1
                report.evidence.append(f"hipcc compile OK: {candidate.path}")
            else:
                report.compile_failed += 1
                tail = (result.stdout or "").splitlines()[-5:]
                report.errors.append(
                    f"hipcc compile failed for {candidate.path}: rc={result.return_code} "
                    + " | ".join(tail)
                )

        if _looks_like_pytorch_extension(Path(self.repo_root)):
            report.evidence.append("pytorch_extension_detected: setup.py mentions CUDAExtension")
            cmd = self.compiler.python_extension_probe()
            try:
                result = self.executor(cmd)
            except Exception as exc:
                report.errors.append(f"python extension probe error: {exc}")
                return
            if result.ok and "KERNEL_MIGRATION_IMPORT_PROBE_OK" in (result.stdout or ""):
                report.evidence.append("pytorch_extension_probe: import OK on ROCm")
            else:
                report.errors.append(
                    f"pytorch extension probe failed: rc={result.return_code}"
                )

    def _finalize_status(
        self,
        report: KernelMigrationReport,
        toolchain: Dict[str, bool],
        candidates: List[KernelCandidate],
    ) -> None:
        if not candidates:
            report.status = "no_kernels"
            report.degradation = "D0"
            return
        # Dry-run is a planning-only mode; treat toolchain probes as moot.
        if self.dry_run:
            report.status = "hipify_planned"
            report.degradation = "D0"
            return
        no_hipify = not (toolchain["hipify_clang"] or toolchain["hipify_perl"])
        no_hipcc = not toolchain["hipcc"]
        if no_hipify and no_hipcc:
            report.status = "unsupported"
            report.degradation = "D4"  # acceleration disabled (toolchain missing)
            return
        if report.kernels_applied == 0:
            report.status = "hipify_planned"
            report.degradation = "D2"
            return
        if no_hipcc:
            # Hipify ran, but we cannot verify compilation.
            report.status = "hipify_applied"
            if report.manual_fix_count > 0:
                report.status = "manual_fixes_required"
            report.degradation = "D2"
            return
        if report.compile_failed == 0 and report.compile_passed > 0:
            report.status = "compile_passed"
            report.degradation = "D1"
            return
        if report.manual_fix_count > 0:
            report.status = "manual_fixes_required"
            report.degradation = "D3"
            return
        report.status = "hipify_applied"
        report.degradation = "D2"

    def _write_report(self, report: KernelMigrationReport) -> Optional[str]:
        try:
            os.makedirs(self.report_dir, exist_ok=True)
            target = os.path.join(self.report_dir, self.REPORT_BASENAME)
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(report.to_dict(), handle, indent=2, default=str)
            report.evidence.append(f"report_written: {target}")
            return target
        except OSError as exc:
            report.errors.append(f"report_write_failed: {exc}")
            return None

    # ── Public entry point ──────────────────────────────────────────────────

    def run(self) -> KernelMigrationReport:
        report = KernelMigrationReport(
            repo_id=self.repo_id,
            attempt_id=self.attempt_id,
            started_at=time.time(),
        )

        try:
            candidates = self._phase_inventory(report)
            if not candidates:
                report.status = "no_kernels"
                report.completed_at = time.time()
                self._write_report(report)
                return report

            toolchain = self._toolchain_available()
            report.evidence.append(
                "toolchain: " + ", ".join(f"{k}={v}" for k, v in toolchain.items())
            )

            if not (toolchain["hipify_clang"] or toolchain["hipify_perl"]) \
                    and not toolchain["hipcc"] and not self.dry_run:
                report.status = "unsupported"
                report.degradation = "D4"
                report.errors.append(
                    "no hipify-clang, hipify-perl, or hipcc found in PATH"
                )
                report.completed_at = time.time()
                self._write_report(report)
                return report

            examine_output = self._phase_examine(report, candidates)
            apply_output = self._phase_apply(report, candidates, examine_output)
            self._phase_granular_fix(report, candidates, examine_output, apply_output)
            self._phase_verify(report, candidates)
            self._finalize_status(report, toolchain, candidates)
        except Exception as exc:
            report.errors.append(f"unhandled_converter_error: {type(exc).__name__}: {exc}")
            if not report.status or report.status == "no_kernels":
                report.status = "manual_fixes_required"

        report.completed_at = time.time()
        self._write_report(report)
        return report


def repo_has_cuda_sources(repo_root: str) -> bool:
    """Quick trigger predicate for the configuration agent.

    True when the repo contains at least one ``.cu`` or ``.cuh`` file outside
    skip dirs. Cheap and side-effect-free — safe to call before deciding
    whether to instantiate ``KernelConverterAgent``.
    """
    root = Path(repo_root)
    if not root.exists():
        return False
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv",
            "build", "dist", "site-packages", "graphify-out", "checkpoints",
            "wandb", "outputs"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
        for name in filenames:
            if name.endswith((".cu", ".cuh")):
                return True
    return False


_CUDA_COMPILE_ERROR_RE = re.compile(
    r"""(?xi)
        \bnvcc\b
      | \bcudaMalloc\b
      | \bcudaMemcpy\b
      | \bcudaError\b
      | identifier\s+['"]?cudaError_t['"]?\s+is\s+undefined
      | cuda_runtime\.h
      | cuda_fp16\.h
      | cublas_v2\.h
      | cusparse\.h
      | curand\.h
      | nccl\.h
      | undefined\s+reference\s+to\s+`?cuda
    """
)


def looks_like_cuda_compile_error(observation: str) -> bool:
    """True when a turn observation strongly suggests a CUDA compile error."""
    if not observation:
        return False
    return bool(_CUDA_COMPILE_ERROR_RE.search(observation))


__all__ = [
    "ALLOWED_SCOPE",
    "FORBIDDEN_ACTIONS",
    "KernelConverterAgent",
    "build_subagent_task_packet",
    "looks_like_cuda_compile_error",
    "repo_has_cuda_sources",
]
