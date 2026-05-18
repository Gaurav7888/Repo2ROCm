"""KB store + error classifier + rule engine."""
from __future__ import annotations

from pathlib import Path

from repo2rocm.learning import KBStore, ClassifiedError, ErrorClassifier
from repo2rocm.learning.kb_store import CompatibilityRecord, Rule
from repo2rocm.learning.rules import RuleEngine, SEED_RULES


def test_kb_compatibility_round_trip(tmp_path: Path):
    kb = KBStore(tmp_path / "kb.sqlite")
    rec = CompatibilityRecord(
        package="torch",
        rocm_version="6.2",
        compatible=True,
        install_method="pip",
        install_commands=["pip install torch"],
        confidence=0.95,
    )
    kb.upsert_compatibility(rec)
    out = kb.get_compatibility("torch")
    assert len(out) == 1
    assert out[0].compatible is True
    kb.close()


def test_error_classifier_matches_seed_patterns():
    cls = ErrorClassifier(kb=None)
    e = cls.classify("Traceback ... ModuleNotFoundError: No module named 'distutils'")
    assert e.error_class == "py312_distutils"
    e2 = cls.classify("AssertionError: Torch not compiled with CUDA enabled")
    assert e2.error_class == "missing_torch_cuda"


def test_rule_engine_matches_when_dict(tmp_path: Path):
    kb = KBStore(tmp_path / "kb.sqlite")
    for r in SEED_RULES:
        kb.upsert_rule(r)
    engine = RuleEngine(kb)
    matches = engine.evaluate({"frameworks": "pytorch", "rocm_mode": True})
    names = [m.rule.name for m in matches]
    assert "prefer_rocm_pytorch_for_torch" in names
    kb.close()
