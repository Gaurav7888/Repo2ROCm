"""
Core data models for the Repo2ROCm intelligence layer.

Every component (trajectory store, KB, rule engine, learning pipeline,
error classifier, DAG executor) shares these types.  Keep them in one
place so serialisation and cross-component references stay consistent.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ── Enums ────────────────────────────────────────────────────────────────────

class BuildOutcome(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


class AgentRole(Enum):
    PLANNER = "planner"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    CODE_PATCH = "code_patch"
    CUDA_KERNEL = "cuda_kernel"
    TRITON_KERNEL = "triton_kernel"
    VERIFICATION = "verification"
    PAPER_REPRODUCTION = "paper_reproduction"
    DOCKER = "docker"


class ActionType(Enum):
    BASH = "bash"
    DIFF = "diff"
    TOOL = "tool"
    REVERT = "revert"
    PLAN = "plan"
    KB_QUERY = "kb_query"
    ERROR_CLASSIFY = "error_classify"
    DETERMINISTIC_FIX = "deterministic_fix"


class ErrorSeverity(Enum):
    FATAL = "fatal"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class RuleSource(Enum):
    EXPERT = "expert"
    LEARNED = "learned"
    PAPER = "paper"
    SEED = "seed"


class KBUpdateType(Enum):
    ADD_FACT = "add_fact"
    DEPRECATE_FIX = "deprecate_fix"
    UPDATE_CONFIDENCE = "update_confidence"
    SUPERSEDE_RULE = "supersede_rule"
    ADD_ERROR_PATTERN = "add_error_pattern"
    ADD_INSTALL_PATH = "add_install_path"
    UPDATE_VERSION_BOUNDS = "update_version_bounds"


class MemoryPhase(Enum):
    BEGIN = "begin"
    IN = "in"


class DAGNodeState(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class KernelPhase(Enum):
    CORRECTNESS = "correctness"
    OPTIMIZATION = "optimization"


# ── Core Data Models ─────────────────────────────────────────────────────────

@dataclass
class TrajectoryRecord:
    """Single action taken by an agent during a build attempt."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    repo_id: str = ""
    attempt_id: str = ""
    agent: str = ""
    timestamp: float = field(default_factory=time.time)
    action_type: str = ""
    action_content: str = ""
    observation_raw: str = ""
    observation_parsed: Optional[Dict[str, Any]] = None
    outcome: str = ""
    return_code: Optional[int] = None
    duration_seconds: float = 0.0
    led_to_success: Optional[bool] = None
    kb_rules_applied: List[str] = field(default_factory=list)
    novel_situation: bool = False
    error_class: Optional[str] = None
    turn_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrajectoryRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BuildFingerprint:
    """Canonical representation of a repo's build characteristics for nearest-neighbor lookup."""
    repo_id: str = ""
    frameworks: Set[str] = field(default_factory=set)
    cuda_deps: Set[str] = field(default_factory=set)
    build_system: str = ""
    python_version: str = ""
    has_custom_cuda_kernels: bool = False
    has_triton_kernels: bool = False
    workload_type: str = ""
    model_scale: str = ""
    top_imports: List[str] = field(default_factory=list)
    config_files_present: List[str] = field(default_factory=list)
    has_distributed: bool = False

    def signature(self) -> str:
        """Deterministic hash for dedup and lookup."""
        canonical = json.dumps({
            "frameworks": sorted(self.frameworks),
            "cuda_deps": sorted(self.cuda_deps),
            "build_system": self.build_system,
            "has_custom_cuda_kernels": self.has_custom_cuda_kernels,
            "has_triton_kernels": self.has_triton_kernels,
            "workload_type": self.workload_type,
            "has_distributed": self.has_distributed,
        }, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["frameworks"] = sorted(self.frameworks)
        d["cuda_deps"] = sorted(self.cuda_deps)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BuildFingerprint:
        d = dict(d)
        d["frameworks"] = set(d.get("frameworks", []))
        d["cuda_deps"] = set(d.get("cuda_deps", []))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def similarity(self, other: BuildFingerprint) -> float:
        """Jaccard-like similarity for nearest-neighbor matching."""
        scores = []
        if self.frameworks and other.frameworks:
            intersection = len(self.frameworks & other.frameworks)
            union = len(self.frameworks | other.frameworks)
            scores.append(intersection / union if union else 0.0)
        if self.cuda_deps and other.cuda_deps:
            intersection = len(self.cuda_deps & other.cuda_deps)
            union = len(self.cuda_deps | other.cuda_deps)
            scores.append(intersection / union if union else 0.0)
        scores.append(1.0 if self.build_system == other.build_system else 0.0)
        scores.append(1.0 if self.workload_type == other.workload_type else 0.0)
        scores.append(1.0 if self.has_custom_cuda_kernels == other.has_custom_cuda_kernels else 0.0)
        scores.append(1.0 if self.has_triton_kernels == other.has_triton_kernels else 0.0)
        scores.append(1.0 if self.has_distributed == other.has_distributed else 0.0)
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class ErrorPattern:
    """A classified error pattern with associated fixes."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signature: str = ""
    error_class: str = ""
    description: str = ""
    regex_pattern: str = ""
    severity: str = ErrorSeverity.ERROR.value
    known_fix_ids: List[str] = field(default_factory=list)
    rocm_version_range: str = ""
    evidence_count: int = 0
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ErrorPattern:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Fix:
    """A concrete fix for an error pattern — executable, not prose."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    commands: List[str] = field(default_factory=list)
    patch_content: Optional[str] = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    success_rate: float = 0.0
    evidence_count: int = 0
    supersedes: List[str] = field(default_factory=list)
    valid_rocm_range: str = ""
    valid_gpu_arch: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Fix:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Rule:
    """Executable rule with condition/action/confidence lifecycle."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    condition: Dict[str, Any] = field(default_factory=dict)
    action: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    source: str = RuleSource.SEED.value
    created_from_attempts: List[str] = field(default_factory=list)
    supersedes: List[str] = field(default_factory=list)
    valid_rocm_range: str = ""
    valid_gpu_arch: List[str] = field(default_factory=list)
    evidence_count: int = 0
    success_rate: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_applied: float = 0.0
    deprecated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Rule:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def record_application(self, success: bool):
        """Update statistics after rule application."""
        self.evidence_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.success_rate = self.success_count / self.evidence_count
        self.last_applied = time.time()
        if self.success_rate < 0.3 and self.evidence_count >= 5:
            self.deprecated = True

    def matches_condition(self, context: Dict[str, Any]) -> bool:
        """Check if this rule's condition matches the given context."""
        for key, expected in self.condition.items():
            actual = context.get(key)
            if actual is None:
                return False
            if key == "error_pattern":
                import re
                if not re.search(expected, str(actual)):
                    return False
            elif key in ("rocm_version_range", "python_version_range"):
                continue  # version range checks handled by caller
            elif isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True


@dataclass
class BuildAttempt:
    """Record of a complete build attempt for a repository."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    repo_id: str = ""
    repo_url: str = ""
    sha: str = ""
    fingerprint: Optional[BuildFingerprint] = None
    outcome: str = BuildOutcome.FAILURE.value
    duration_minutes: float = 0.0
    docker_image: str = ""
    rocm_version: str = ""
    gpu_arch: str = ""
    total_turns: int = 0
    total_tokens: int = 0
    rules_applied: List[str] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)
    errors_encountered: List[str] = field(default_factory=list)
    novel_errors: List[str] = field(default_factory=list)
    trajectory_file: str = ""
    dockerfile_path: str = ""
    plan_text: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.fingerprint:
            d["fingerprint"] = self.fingerprint.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BuildAttempt:
        d = dict(d)
        fp = d.pop("fingerprint", None)
        attempt = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if fp and isinstance(fp, dict):
            attempt.fingerprint = BuildFingerprint.from_dict(fp)
        return attempt


@dataclass
class KBUpdateProposal:
    """Proposed update to the knowledge base from the learning pipeline."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    update_type: str = KBUpdateType.ADD_FACT.value
    target_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    source_attempt_id: str = ""
    confidence: float = 0.5
    conflicts_with: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryRequest:
    """Request for memory retrieval during a build session."""
    query: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    phase: str = MemoryPhase.BEGIN.value
    fingerprint: Optional[BuildFingerprint] = None
    current_error: Optional[str] = None
    turn_number: int = 0


@dataclass
class MemoryItem:
    """A retrieved memory item (rule, fix, or guidance)."""
    id: str = ""
    content: str = ""
    item_type: str = ""  # "rule", "fix", "pattern", "guidance"
    confidence: float = 0.0
    source_rule_id: Optional[str] = None
    source_fix_id: Optional[str] = None
    executable: bool = False
    commands: List[str] = field(default_factory=list)


@dataclass
class MemoryResponse:
    """Response containing retrieved memories for an agent turn."""
    items: List[MemoryItem] = field(default_factory=list)
    deterministic_fixes: List[Fix] = field(default_factory=list)
    guidance_text: str = ""
    confidence: float = 0.0


# ── DAG Models ───────────────────────────────────────────────────────────────

@dataclass
class DAGNode:
    """A node in the execution DAG."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    agent_role: str = AgentRole.CONFIGURATION.value
    commands: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    state: str = DAGNodeState.PENDING.value
    can_parallel: bool = True
    estimated_duration_minutes: float = 5.0
    failure_probability: float = 0.1
    critical_path: bool = False
    fallback_node_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: float = 0.0
    completed_at: float = 0.0
    container_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionDAG:
    """Directed Acyclic Graph of tasks for parallel execution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    repo_id: str = ""
    nodes: Dict[str, DAGNode] = field(default_factory=dict)
    replan_count: int = 0
    max_replans: int = 3

    def add_node(self, node: DAGNode) -> str:
        self.nodes[node.id] = node
        return node.id

    def get_ready_nodes(self) -> List[DAGNode]:
        """Return nodes whose dependencies are all satisfied."""
        ready = []
        for node in self.nodes.values():
            if node.state != DAGNodeState.PENDING.value:
                continue
            deps_satisfied = all(
                self.nodes[dep_id].state == DAGNodeState.SUCCESS.value
                for dep_id in node.dependencies
                if dep_id in self.nodes
            )
            if deps_satisfied:
                ready.append(node)
        return ready

    def get_critical_path(self) -> List[str]:
        """Compute critical path (longest chain) through the DAG."""
        memo: Dict[str, float] = {}

        def _longest(node_id: str) -> float:
            if node_id in memo:
                return memo[node_id]
            node = self.nodes[node_id]
            if not node.dependencies:
                memo[node_id] = node.estimated_duration_minutes
                return memo[node_id]
            max_dep = max(_longest(d) for d in node.dependencies if d in self.nodes)
            memo[node_id] = max_dep + node.estimated_duration_minutes
            return memo[node_id]

        if not self.nodes:
            return []

        for nid in self.nodes:
            _longest(nid)

        path = []
        sorted_nodes = sorted(memo.items(), key=lambda x: -x[1])
        for nid, _ in sorted_nodes:
            node = self.nodes[nid]
            node.critical_path = True
            path.append(nid)
        return path

    def is_complete(self) -> bool:
        return all(
            n.state in (DAGNodeState.SUCCESS.value, DAGNodeState.FAILED.value, DAGNodeState.SKIPPED.value)
            for n in self.nodes.values()
        )

    def has_failed_critical(self) -> bool:
        return any(
            n.state == DAGNodeState.FAILED.value and n.critical_path
            for n in self.nodes.values()
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "replan_count": self.replan_count,
        }


# ── Paper Extraction Models ─────────────────────────────────────────────────

@dataclass
class PaperMetadata:
    """Structured extraction from a research paper."""
    arxiv_id: str = ""
    title: str = ""
    hardware_used: List[str] = field(default_factory=list)
    cuda_version_mentioned: str = ""
    key_libraries: List[str] = field(default_factory=list)
    custom_kernels_described: bool = False
    kernel_purpose: List[str] = field(default_factory=list)
    reproduction_commands: List[str] = field(default_factory=list)
    benchmark_metrics: Dict[str, str] = field(default_factory=dict)
    model_scale: str = ""
    training_compute: str = ""
    key_tricks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PaperMetadata:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class KernelInfo:
    """Information about a detected CUDA or Triton kernel."""
    file_path: str = ""
    kernel_type: str = ""  # "cuda" or "triton"
    purpose: str = ""
    dependencies: List[str] = field(default_factory=list)
    hipified: bool = False
    hipify_issues: List[str] = field(default_factory=list)
    numerically_verified: bool = False
    optimized: bool = False
    autotune_configs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ReproductionResult:
    """Result of a paper reproduction attempt."""
    command: str = ""
    expected_output: str = ""
    actual_output: str = ""
    match_status: str = ""  # "match", "partial", "mismatch"
    metric_deltas: Dict[str, float] = field(default_factory=dict)
    scaled: bool = False
    scale_factor: float = 1.0


@dataclass
class ExperimentCandidate:
    """A candidate paper experiment considered for reproduction."""
    name: str = ""
    section: str = ""
    expected_metric_name: str = ""
    expected_metric_value: str = ""
    expected_metric_units: str = ""
    hardware: str = ""
    est_runtime_minutes: float = 0.0
    runtime_bucket: str = ""  # "small" | "medium" | "large"
    paper_config: Dict[str, Any] = field(default_factory=dict)
    suggested_command: str = ""
    code_available: bool = False
    matched_files: List[str] = field(default_factory=list)
    tolerance_rule: str = ""  # free-form, e.g. "<=15% for speedups"
    notes: str = ""
    rank_score: float = 0.0
    # New fields for smarter ranking (all optional, default-preserving).
    metric_class: str = ""  # ratio_speedup | accuracy | quality | absolute_perf | other
    is_baseline: bool = False  # True if this looks like a no-method baseline row
    # Precise configuration extracted from the paper/README (all non-default
    # flags the entry script needs to exhibit the paper's reported metric).
    caveats: List[str] = field(default_factory=list)
    # Flags from paper_config that the repo's entry script does NOT expose as
    # CLI args — the runtime agent must work around these (code-edit or skip).
    missing_flags: List[str] = field(default_factory=list)
    # Short citation of where in the paper/README the config was found.
    config_source: str = ""
    # Repo config files (yaml/toml/json/cfg) that govern this experiment's
    # hyperparameters. The runtime agent should read/override these instead of
    # guessing values when the paper is ambiguous.
    codebase_config_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentCandidate":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
