"""Docker Hub MCP server — exposes `list_tags`, `get_manifest`.

Run with `python -m repo2rocm.mcp.servers.docker_hub` over stdio.
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import httpx


async def list_tags(image: str, limit: int = 20) -> dict[str, Any]:
    repo = image if "/" in image else f"library/{image}"
    url = f"https://hub.docker.com/v2/repositories/{repo}/tags?page_size={limit}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    return {
        "image": repo,
        "tags": [t["name"] for t in data.get("results", [])],
    }


# Minimal JSON-RPC 2.0 stdio server.
async def main() -> None:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(loop=loop)
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = req.get("method")
        params = req.get("params") or {}
        rid = req.get("id")
        try:
            if method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": "list_tags",
                            "description": "List Docker Hub tags for an image.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "image": {"type": "string"},
                                    "limit": {"type": "integer"},
                                },
                                "required": ["image"],
                            },
                        }
                    ]
                }
            elif method == "tools/call":
                tname = params.get("name")
                args = params.get("arguments") or {}
                if tname == "list_tags":
                    result = await list_tags(**args)
                else:
                    raise RuntimeError(f"unknown tool: {tname}")
            else:
                raise RuntimeError(f"unknown method: {method}")
            resp = {"jsonrpc": "2.0", "id": rid, "result": result}
        except Exception as exc:
            resp = {
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": -32000, "message": str(exc)},
            }
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
