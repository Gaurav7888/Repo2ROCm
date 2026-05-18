"""MCP (Model Context Protocol) — universal tool protocol.

We ship two reference servers (DockerHub, PyPI) and a stdio transport. The
servers are usable from any MCP client, not just Repo2ROCm.

If the optional `mcp` package is not installed, the servers degrade to a no-op
import; the equivalent functionality is still available via the in-process
tools (`tools/external/`).
"""
from repo2rocm.mcp.client import MCPClient, MCPClientError

__all__ = ["MCPClient", "MCPClientError"]
