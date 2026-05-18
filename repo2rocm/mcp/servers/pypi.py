"""PyPI MCP server — `versions`, `classifiers`, `requires_python`."""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import httpx


async def versions(package: str, limit: int = 12) -> dict[str, Any]:
    url = f"https://pypi.org/pypi/{package}/json"
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url)
        if r.status_code == 404:
            return {"package": package, "versions": [], "not_found": True}
        r.raise_for_status()
        data = r.json()
    releases = data.get("releases") or {}
    vs = sorted(releases.keys(), reverse=True)[:limit]
    return {
        "package": package,
        "versions": vs,
        "requires_python": (data.get("info") or {}).get("requires_python"),
        "classifiers": ((data.get("info") or {}).get("classifiers") or [])[:20],
    }


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
                            "name": "versions",
                            "description": "Query PyPI versions + classifiers.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "package": {"type": "string"},
                                    "limit": {"type": "integer"},
                                },
                                "required": ["package"],
                            },
                        }
                    ]
                }
            elif method == "tools/call":
                tname = params.get("name")
                args = params.get("arguments") or {}
                if tname == "versions":
                    result = await versions(**args)
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
