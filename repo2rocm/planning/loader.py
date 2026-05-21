"""WorkflowTemplate loader. Reads YAML phase definitions packaged under workflows/."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


@dataclass
class WorkflowPhase:
    id: str
    name: str
    purpose: str
    agent: str
    skills: list[str] = field(default_factory=list)
    success_marker: str = ""
    parallel_safe: bool = False
    # `True` only for the LAST step in reproduce mode — anything else is a
    # mid-flight checkpoint. The planner agent uses this to mark the
    # corresponding PlanStep's success_marker as the terminal one.
    terminal: bool = False


@dataclass
class WorkflowTemplate:
    mode: str
    description: str
    phases: list[WorkflowPhase]

    def to_yaml(self) -> str:
        """Render back to YAML (compact, no-dependencies fallback if PyYAML missing)."""
        if yaml is None:
            return self._fallback_render()
        return yaml.safe_dump(
            {
                "mode": self.mode,
                "description": self.description.strip(),
                "phases": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "purpose": p.purpose,
                        "agent": p.agent,
                        "skills": p.skills,
                        "success_marker": p.success_marker,
                        "parallel_safe": p.parallel_safe,
                        "terminal": p.terminal,
                    }
                    for p in self.phases
                ],
            },
            sort_keys=False,
        )

    def _fallback_render(self) -> str:
        lines = [f"mode: {self.mode}", "description: |", f"  {self.description.strip()}", "phases:"]
        for p in self.phases:
            lines.append(f"  - id: {p.id}")
            lines.append(f"    name: {p.name}")
            lines.append(f"    purpose: {p.purpose}")
            lines.append(f"    agent: {p.agent}")
            if p.skills:
                lines.append(f"    skills: [{', '.join(p.skills)}]")
            if p.success_marker:
                lines.append(f"    success_marker: {p.success_marker}")
        return "\n".join(lines)


_WORKFLOWS_DIR = Path(__file__).parent / "workflows"


def load_workflow(mode: str) -> WorkflowTemplate:
    if mode not in ("functional", "reproduce"):
        raise ValueError(f"mode must be 'functional' or 'reproduce', got {mode!r}")
    path = _WORKFLOWS_DIR / f"{mode}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"workflow not found: {path}")
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        # Minimal fallback parser — production should have PyYAML; we degrade
        # gracefully in test environments without it.
        return _parse_yaml_fallback(text, mode=mode)
    data = yaml.safe_load(text) or {}
    return _from_dict(data)


def _from_dict(data: dict[str, Any]) -> WorkflowTemplate:
    phases = []
    for p in data.get("phases", []) or []:
        phases.append(
            WorkflowPhase(
                id=str(p["id"]),
                name=str(p["name"]),
                purpose=str(p.get("purpose", "")),
                agent=str(p["agent"]),
                skills=list(p.get("skills", []) or []),
                success_marker=str(p.get("success_marker", "") or p.get("success", "") or ""),
                parallel_safe=bool(p.get("parallel_safe", False)),
                terminal=bool(p.get("terminal", False)),
            )
        )
    return WorkflowTemplate(
        mode=str(data.get("mode", "")),
        description=str(data.get("description", "")),
        phases=phases,
    )


def _parse_yaml_fallback(text: str, *, mode: str) -> WorkflowTemplate:
    """Minimal best-effort YAML parser used only when PyYAML is unavailable."""
    phases: list[WorkflowPhase] = []
    current: dict[str, Any] = {}
    in_phases = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("phases:"):
            in_phases = True
            continue
        if not in_phases:
            continue
        if line.startswith("  - id:"):
            if current:
                phases.append(_phase_from(current))
            current = {"id": line.split(":", 1)[1].strip()}
        elif current and line.startswith("    "):
            k, _, v = line.strip().partition(":")
            current[k.strip()] = v.strip().strip("[]").strip()
    if current:
        phases.append(_phase_from(current))
    return WorkflowTemplate(mode=mode, description="", phases=phases)


def _phase_from(d: dict[str, Any]) -> WorkflowPhase:
    skills_raw = d.get("skills", "")
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()]
    return WorkflowPhase(
        id=d.get("id", ""),
        name=d.get("name", ""),
        purpose=d.get("purpose", ""),
        agent=d.get("agent", ""),
        skills=skills,
        success_marker=d.get("success_marker", "") or d.get("success", ""),
        parallel_safe=str(d.get("parallel_safe", "false")).lower() == "true",
    )
