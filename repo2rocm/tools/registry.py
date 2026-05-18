"""Assemble the tool pool sent to the model for a specific agent.

Applies tool-restriction filters (allowed/disallowed) and converts each tool to a
provider-agnostic `ToolSpec` with the LAST entry getting `cache_control: ephemeral`
to anchor the prompt-cache breakpoint after the built-in tool list.
"""
from __future__ import annotations

from typing import Iterable

from repo2rocm.core.api import ToolSpec
from repo2rocm.tools.base import BaseTool, get_all_tools


def assemble_tool_pool(
    *,
    allowed: Iterable[str] | None = None,
    disallowed: Iterable[str] | None = None,
    extra: Iterable[BaseTool] = (),
) -> tuple[list[BaseTool], list[ToolSpec]]:
    """Return (tools, specs). Tools sorted alphabetically (cache stability)."""
    all_tools = {t.name: t for t in get_all_tools()}
    for t in extra:
        all_tools[t.name] = t

    allowed_set = set(allowed) if allowed else None
    disallowed_set = set(disallowed) if disallowed else set()

    selected: list[BaseTool] = []
    for name in sorted(all_tools):
        if allowed_set is not None and name not in allowed_set:
            continue
        if name in disallowed_set:
            continue
        selected.append(all_tools[name])

    specs = [
        ToolSpec(
            name=t.name,
            description=t.description[:2048],  # cap per Ch.15 MCP wisdom
            input_schema=t.schema(),
        )
        for t in selected
    ]
    if specs:
        specs[-1].cache_control = {"type": "ephemeral"}
    return selected, specs
