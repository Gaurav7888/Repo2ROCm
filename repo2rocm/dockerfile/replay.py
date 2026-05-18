"""Run `docker build` on the synthesized Dockerfile and verify it succeeds."""
from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VerifyResult:
    succeeded: bool
    image_id: str
    log_tail: str


async def verify_dockerfile(
    dockerfile_dir: Path, *, tag: str = "repo2rocm-verified", timeout_s: float = 1800.0
) -> VerifyResult:
    if shutil.which("docker") is None:
        return VerifyResult(False, "", "docker CLI not available")
    proc = await asyncio.create_subprocess_exec(
        "docker", "build", "-t", tag, ".",
        cwd=str(dockerfile_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        return VerifyResult(False, "", "docker build timed out")
    out = stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        return VerifyResult(False, "", out[-4000:])
    return VerifyResult(True, tag, out[-2000:])
