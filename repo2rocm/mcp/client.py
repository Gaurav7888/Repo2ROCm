"""Thin MCP client wrapper. Stub-friendly when `mcp` package missing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class MCPClientError(Exception):
    pass


@dataclass
class MCPClient:
    name: str
    transport: str = "stdio"  # stdio | http | inprocess
    command: list[str] = field(default_factory=list)
    url: str = ""

    async def list_tools(self) -> list[dict[str, Any]]:
        try:
            import mcp  # noqa: F401
        except ImportError as exc:
            raise MCPClientError("mcp package not installed; pip install repo2rocm[mcp]") from exc
        # Production: open the mcp.ClientSession and call tools/list
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            import mcp  # noqa: F401
        except ImportError as exc:
            raise MCPClientError("mcp package not installed") from exc
        return {"unimplemented": True, "tool": name, "args": arguments}
