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

__all__ = [
    "CommandResult",
    "FixSuggestion",
    "KernelCandidate",
    "KernelMigrationAgent",
    "KernelMigrationReport",
    "MigrationStage",
    "discover_cuda_sources",
]
