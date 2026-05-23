"""Correctness-first CUDA/Triton kernel migration scaffolding."""

from .scaffold import (
    CommandResult,
    FixSuggestion,
    KernelCandidate,
    KernelMigrationAgent,
    KernelMigrationReport,
    MigrationStage,
    discover_cuda_sources,
)
from .executor_adapter import (
    DryRunExecutor,
    SandboxExecutor,
    make_executor,
)

__all__ = [
    "CommandResult",
    "FixSuggestion",
    "KernelCandidate",
    "KernelMigrationAgent",
    "KernelMigrationReport",
    "MigrationStage",
    "discover_cuda_sources",
    "DryRunExecutor",
    "SandboxExecutor",
    "make_executor",
]
