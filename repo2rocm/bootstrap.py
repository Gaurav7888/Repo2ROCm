"""Bootstrap — the 5-phase init pipeline.

Phases (analogue of Ch. 2 of the Claude Code book):
  1. Config load
  2. Observability setup (tracing + metrics endpoint + logging)
  3. Hooks snapshot
  4. Skill discovery (frontmatter only)
  5. Tool registration

Bootstrap is idempotent and safe to call multiple times.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from repo2rocm.config import Settings, get_settings
from repo2rocm.core.api import AMDGatewayClient, AnthropicClient, ModelClient, MockClient
from repo2rocm.core.hooks import HooksSnapshot, capture_hooks_snapshot, register_builtin_hooks
from repo2rocm.core.hooks.builtin import GateState
from repo2rocm.observability import (
    Metrics,
    METRICS,
    checkpoint,
    setup_observability,
)
from repo2rocm.skills import SkillCatalog, discover_skills

_lock = threading.Lock()
_done = False


@dataclass
class Bootstrap:
    settings: Settings
    hooks_snapshot: HooksSnapshot
    skill_catalog: SkillCatalog
    gate_state: GateState
    metrics: Metrics

    def make_client(self, model: str | None = None) -> ModelClient:
        s = self.settings
        m = model or s.llm_model
        if s.llm_provider == "mock":
            return MockClient(model=m)
        if s.llm_provider == "amd_gateway":
            # Pick the right gateway endpoint based on the model name:
            #   Claude models live at /claude3 (model-in-path, Anthropic shape)
            #   GPT-oss / Llama / Qwen / DeepSeek live at /OnPrem (model-in-body, OpenAI shape)
            base_url = s.amd_gateway_base_url
            if not base_url:
                base_url = (
                    "https://llm-api.amd.com/claude3"
                    if "claude" in m.lower()
                    else "https://llm-api.amd.com/OnPrem"
                )
            return AMDGatewayClient(model=m, api_key=s.llm_api_key(), base_url=base_url)
        return AnthropicClient(model=m, api_key=s.llm_api_key())


_bootstrap: Bootstrap | None = None


def bootstrap(*, force: bool = False) -> Bootstrap:
    global _done, _bootstrap

    with _lock:
        if _done and not force and _bootstrap is not None:
            return _bootstrap

        checkpoint("bootstrap.start")

        settings = get_settings()
        checkpoint("bootstrap.config_load")

        setup_observability(
            service_name="repo2rocm",
            otlp_endpoint=settings.otel_endpoint or None,
        )
        METRICS.start_http_endpoint(port=settings.metrics_port)
        checkpoint("bootstrap.observability")

        snap = capture_hooks_snapshot()
        gate = GateState()
        register_builtin_hooks(snap, gate)
        checkpoint("bootstrap.hooks_snapshot")

        skills = discover_skills()
        checkpoint("bootstrap.skills_discovered")

        # Register all tools (idempotent)
        from repo2rocm.tools.repo import register_repo_tools
        from repo2rocm.tools.docker import register_docker_tools
        from repo2rocm.tools.packaging import register_packaging_tools
        from repo2rocm.tools.external import register_external_tools
        from repo2rocm.tools.verify import register_verify_tools
        from repo2rocm.tools.agent_tool import register_agent_tools
        from repo2rocm.tools.skills import register_skill_tools
        from repo2rocm.tools.planning import register_planning_tools
        from repo2rocm.tools.paper import register_paper_tools

        register_repo_tools()
        register_docker_tools()
        register_packaging_tools()
        register_external_tools()
        register_verify_tools()
        register_agent_tools()
        register_skill_tools()
        register_planning_tools()
        register_paper_tools()
        checkpoint("bootstrap.tools_registered")

        _bootstrap = Bootstrap(
            settings=settings,
            hooks_snapshot=snap,
            skill_catalog=skills,
            gate_state=gate,
            metrics=METRICS,
        )
        _done = True
        checkpoint("bootstrap.done")
        return _bootstrap


def reset_for_tests() -> None:
    global _done, _bootstrap
    _done = False
    _bootstrap = None
