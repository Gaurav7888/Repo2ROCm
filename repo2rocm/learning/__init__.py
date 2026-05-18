"""Learning layer — slim, structured KB + trajectory store + distiller.

We keep ONLY structured knowledge that transfers between repos:
  * package compatibility / install paths (Compatibility table)
  * deterministic rules (RuleEngine)
  * error patterns (ErrorClassifier)

Free-form natural-language "lessons" are intentionally dropped — they overfit.
"""
from repo2rocm.learning.kb_store import KBStore
from repo2rocm.learning.trajectory_store import TrajectoryStore, BuildAttempt
from repo2rocm.learning.error_classifier import ErrorClassifier, ClassifiedError
from repo2rocm.learning.distiller import TrajectoryDistiller

__all__ = [
    "KBStore",
    "TrajectoryStore",
    "BuildAttempt",
    "ErrorClassifier",
    "ClassifiedError",
    "TrajectoryDistiller",
]
