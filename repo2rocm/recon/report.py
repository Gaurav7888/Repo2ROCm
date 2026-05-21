"""Typed output of the recon pipeline."""
from __future__ import annotations

from pydantic import BaseModel, Field


class FilteredRequirements(BaseModel):
    install: list[str] = Field(default_factory=list)
    skip_preinstalled: list[str] = Field(default_factory=list)
    skip_banned: list[str] = Field(default_factory=list)
    special_handling: list[str] = Field(
        default_factory=list,
        description="Packages with non-pip install recipes (flash-attn, bitsandbytes, ...).",
    )


class Hazard(BaseModel):
    kind: str
    file: str
    line: int = 0
    description: str
    fix: str = ""


class ImageSelection(BaseModel):
    image: str
    tag: str
    workload: str
    score: float = 0.0
    reasoning: list[str] = Field(default_factory=list)


class ReconReport(BaseModel):
    """Everything the planner needs to produce a plan, computed deterministically."""

    repo: str
    sha: str = ""
    mode: str  # "functional" | "reproduce"
    repo_path: str

    framework: str = "unknown"
    python_version: str = ""

    top_imports: list[tuple[str, int]] = Field(default_factory=list)
    cuda_deps: list[str] = Field(default_factory=list)

    config_files: list[str] = Field(default_factory=list)
    install_mechanisms: list[str] = Field(default_factory=list)

    entry_scripts: list[str] = Field(default_factory=list)
    readme_run_commands: list[str] = Field(default_factory=list)
    expected_outcomes: list[str] = Field(default_factory=list)

    image_selection: ImageSelection | None = None
    filtered_requirements: FilteredRequirements = Field(default_factory=FilteredRequirements)

    py312_issues: list[Hazard] = Field(default_factory=list)
    pin_hazards: list[Hazard] = Field(default_factory=list)
    code_hazards: list[Hazard] = Field(default_factory=list)
    training_params: list[Hazard] = Field(default_factory=list)

    top_level: list[str] = Field(default_factory=list)
    readme_excerpt: str = ""

    # Optional paper context (filled by paper-research agent in reproduce mode).
    paper_arxiv_id: str = ""
    paper_title: str = ""

    def render_for_planner(self) -> str:
        """Compact, human-readable summary injected into the planner's system prompt."""
        lines: list[str] = []
        lines.append(f"## Repo: {self.repo}  (mode={self.mode})")
        lines.append(f"Framework: {self.framework}    Python: {self.python_version or '(unspecified)'}")
        if self.image_selection:
            sel = self.image_selection
            lines.append(f"Recommended image: {sel.image}:{sel.tag}  ({sel.workload})")
            for r in sel.reasoning[:5]:
                lines.append(f"  - {r}")
        lines.append("")
        lines.append("Top imports: " + ", ".join(p for p, _ in self.top_imports[:15]))
        if self.cuda_deps:
            lines.append("CUDA deps detected: " + ", ".join(self.cuda_deps))
        lines.append("")
        lines.append("Config files: " + ", ".join(self.config_files))
        lines.append("Install mechanisms: " + ", ".join(self.install_mechanisms))
        lines.append("")

        fr = self.filtered_requirements
        if fr.install:
            lines.append(f"INSTALL ({len(fr.install)}): " + ", ".join(fr.install[:30]))
        if fr.skip_preinstalled:
            lines.append(f"SKIP preinstalled ({len(fr.skip_preinstalled)}): " + ", ".join(fr.skip_preinstalled))
        if fr.skip_banned:
            lines.append(f"SKIP banned ({len(fr.skip_banned)}): " + ", ".join(fr.skip_banned))
        if fr.special_handling:
            lines.append(f"SPECIAL handling ({len(fr.special_handling)}): " + ", ".join(fr.special_handling))
        lines.append("")

        if self.entry_scripts:
            lines.append("Entry scripts: " + ", ".join(self.entry_scripts[:5]))
        if self.readme_run_commands:
            lines.append("README run commands (use these for verification):")
            for c in self.readme_run_commands[:6]:
                lines.append(f"  $ {c}")
        if self.expected_outcomes:
            lines.append("Expected outcomes:")
            for o in self.expected_outcomes[:4]:
                lines.append(f"  - {o}")
        lines.append("")

        if self.py312_issues:
            lines.append(f"Python-3.12 issues: {len(self.py312_issues)}")
            for h in self.py312_issues[:3]:
                lines.append(f"  - {h.file}:{h.line} {h.description}")
        if self.pin_hazards:
            lines.append(f"Pin hazards: {len(self.pin_hazards)}")
            for h in self.pin_hazards[:3]:
                lines.append(f"  - {h.description}")
        if self.code_hazards:
            lines.append(f"Code hazards: {len(self.code_hazards)}")
            for h in self.code_hazards[:3]:
                lines.append(f"  - {h.file}:{h.line} {h.description}")
        if self.training_params:
            lines.append(f"Large training params: {len(self.training_params)}")
        return "\n".join(lines)
