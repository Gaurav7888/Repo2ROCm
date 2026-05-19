"""Setuptools entry point for ``repo2rocm``.

Project metadata lives here (rather than in ``pyproject.toml``'s
``[project]`` table) so that the package installs cleanly on hosts that
ship pre-PEP-621 setuptools (e.g. Ubuntu 20.04 / 22.04 with the system
``setuptools 59.6.0``). Modern toolchains will still pick everything up
through the standard ``setuptools.build_meta`` backend declared in
``pyproject.toml``.
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
LONG_DESCRIPTION = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="repo2rocm",
    version="2.0.0a1",
    description=(
        "Production-grade multi-agent CUDA->ROCm migration system (v2 redesign)."
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Repo2ROCm contributors",
    license="Apache-2.0",
    url="https://github.com/repo2rocm/repo2rocm",
    project_urls={
        "Homepage": "https://github.com/repo2rocm/repo2rocm",
        "Documentation": "https://github.com/repo2rocm/repo2rocm/tree/main/docs",
        "Issues": "https://github.com/repo2rocm/repo2rocm/issues",
    },
    keywords=["llm", "agent", "rocm", "amd", "docker", "migration"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Software Development :: Code Generators",
    ],
    python_requires=">=3.10",
    packages=find_packages(
        include=["repo2rocm", "repo2rocm.*"],
        exclude=["tests", "tests.*", "docs", "examples", "work", "output"],
    ),
    include_package_data=True,
    package_data={
        "repo2rocm": [
            "skills/builtin/*/SKILL.md",
            "skills/builtin/*/*.md",
            "py.typed",
        ],
    },
    install_requires=[
        "pydantic>=2.6,<3",
        "pydantic-settings>=2.2",
        "httpx>=0.27",
        "anthropic>=0.34",
        "tenacity>=8.2",
        "typer>=0.12",
        "rich>=13.7",
        "python-frontmatter>=1.1",
        "docker>=7.0",
        "opentelemetry-api>=1.25",
        "opentelemetry-sdk>=1.25",
        "opentelemetry-exporter-otlp-proto-http>=1.25",
        "prometheus-client>=0.20",
        "structlog>=24.1",
    ],
    extras_require={
        "openai": ["openai>=1.30"],
        "mcp": ["mcp>=1.0"],
        "dev": [
            "pytest>=8.0",
            "pytest-asyncio>=0.23",
            "pytest-cov>=5.0",
            "mypy>=1.10",
            "ruff>=0.5",
            "types-requests",
        ],
    },
    entry_points={
        "console_scripts": [
            "repo2rocm=repo2rocm.cli:app",
        ],
    },
)
