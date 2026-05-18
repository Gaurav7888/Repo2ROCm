"""Self-describing tools. Each tool subclasses BaseTool, declares a Pydantic input model,
and gets JSON-schema, validation, dispatch, permissions and result budgeting for free.
"""
from repo2rocm.tools.base import (
    BaseTool,
    ToolResult,
    ToolUseContext,
    register_tool,
    get_all_tools,
    get_tool,
)
from repo2rocm.tools.registry import assemble_tool_pool

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolUseContext",
    "register_tool",
    "get_all_tools",
    "get_tool",
    "assemble_tool_pool",
]
