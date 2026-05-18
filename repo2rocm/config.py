"""Application config. Pydantic Settings — read from env / dotenv / CLI."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REPO2ROCM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (provider auto-detected if not set: see get_settings)
    llm_provider: str = Field("", description="anthropic | amd_gateway | openai | mock (auto if empty)")
    llm_model: str = Field("", description="defaulted per-provider if empty")
    llm_max_tokens: int = 8_192
    anthropic_api_key: str = ""
    amd_api_key: str = ""
    amd_gateway_base_url: str = Field(
        "",
        description=(
            "Override AMD gateway URL. Defaults: /claude3 for Claude models, "
            "/OnPrem for everything else (auto-routed by model name)."
        ),
    )

    # Filesystem
    root_dir: Path = Field(Path.cwd())
    cache_dir: Path = Field(Path.home() / ".repo2rocm")
    kb_path: Path = Field(Path.home() / ".repo2rocm" / "kb.sqlite")
    trajectories_path: Path = Field(Path.home() / ".repo2rocm" / "trajectories.sqlite")

    # Observability
    otel_endpoint: str = ""
    metrics_port: int = 9464
    log_json: bool | None = None  # None = auto

    # Behavior
    rocm_mode: bool = True
    max_workers_in_flight: int = 3
    enable_streaming: bool = True
    enable_speculative_tools: bool = True
    enable_prompt_cache: bool = True

    def llm_api_key(self) -> str:
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if self.llm_provider == "amd_gateway":
            return self.amd_api_key or os.environ.get("AMD_LLM_API_KEY", "")
        return ""


_DEFAULT_MODELS = {
    "anthropic": "claude-3-5-sonnet-20241022",
    "amd_gateway": "claude-sonnet-4",  # per AMD gateway reference example
    "openai": "gpt-4o-2024-08-06",
    "mock": "mock-model",
}


def _auto_detect_provider() -> str:
    """If REPO2ROCM_LLM_PROVIDER isn't set, pick one based on which API key is present."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("AMD_LLM_API_KEY"):
        return "amd_gateway"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "mock"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        s = Settings()
        if not s.llm_provider:
            s.llm_provider = _auto_detect_provider()
        if not s.llm_model:
            s.llm_model = _DEFAULT_MODELS.get(s.llm_provider, "claude-sonnet-4")
        _settings = s
    return _settings


def reset_settings_for_tests() -> None:
    global _settings
    _settings = None
