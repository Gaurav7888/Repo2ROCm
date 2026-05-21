"""End-to-end recon orchestration."""
from __future__ import annotations

from pathlib import Path

from repo2rocm.recon.configs import (
    detect_install_mechanisms,
    detect_python_version,
    read_config_files,
)
from repo2rocm.recon.cuda import detect_cuda_deps
from repo2rocm.recon.files import find_python_files, list_top_level, read_readme
from repo2rocm.recon.hazards import (
    detect_code_hazards,
    detect_pin_hazards,
    detect_py312_issues,
    detect_training_params,
)
from repo2rocm.recon.image_select import select_rocm_image
from repo2rocm.recon.imports import detect_framework, extract_imports
from repo2rocm.recon.readme import (
    extract_expected_outcomes,
    extract_run_commands,
    find_entry_scripts,
)
from repo2rocm.recon.report import ReconReport
from repo2rocm.recon.requirements_filter import filter_requirements


def run_recon(
    *,
    repo_path: Path | str,
    repo_full_name: str,
    mode: str,
    sha: str = "",
    rocm_base_image_override: str = "",
    max_py_files: int = 500,
) -> ReconReport:
    """Synchronous, no-LLM recon pipeline. Returns a typed `ReconReport`."""
    if mode not in ("functional", "reproduce"):
        raise ValueError(f"mode must be 'functional' or 'reproduce', got {mode!r}")

    repo_path = Path(repo_path)
    config_contents = read_config_files(repo_path)
    readme_name, readme_text = read_readme(repo_path)
    py_files = find_python_files(repo_path, limit=max_py_files)

    import_counts = extract_imports(py_files)
    framework = detect_framework(import_counts)
    python_version = detect_python_version(config_contents)
    install_mechanisms = detect_install_mechanisms(config_contents)
    cuda_deps = detect_cuda_deps(import_counts, config_contents)

    top_imports = sorted(import_counts.items(), key=lambda kv: -kv[1])[:30]

    entry_scripts = find_entry_scripts(str(repo_path), readme_text)
    run_commands = extract_run_commands(readme_text)
    expected_outcomes = extract_expected_outcomes(readme_text)

    # Hazards
    py312 = detect_py312_issues(py_files, str(repo_path))
    pin_hazards = detect_pin_hazards(config_contents)
    code_hazards = detect_code_hazards(py_files, str(repo_path))
    training_params = detect_training_params(py_files, str(repo_path))

    # Image selection — deterministic. Allow CLI override.
    if rocm_base_image_override:
        selection = None  # caller will set base_image directly
        base_for_filter = rocm_base_image_override
    else:
        selection = select_rocm_image(
            import_counts=import_counts,
            config_contents=config_contents,
            readme_text=readme_text,
        )
        base_for_filter = selection.image

    fr = filter_requirements(config_contents=config_contents, base_image=base_for_filter)

    return ReconReport(
        repo=repo_full_name,
        sha=sha,
        mode=mode,
        repo_path=str(repo_path),
        framework=framework,
        python_version=python_version,
        top_imports=top_imports,
        cuda_deps=cuda_deps,
        config_files=list(config_contents.keys()),
        install_mechanisms=install_mechanisms,
        entry_scripts=entry_scripts,
        readme_run_commands=run_commands,
        expected_outcomes=expected_outcomes,
        image_selection=selection,
        filtered_requirements=fr,
        py312_issues=py312,
        pin_hazards=pin_hazards,
        code_hazards=code_hazards,
        training_params=training_params,
        top_level=list_top_level(repo_path),
        readme_excerpt=(readme_text or "")[:4000],
    )
