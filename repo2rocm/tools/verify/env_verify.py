"""EnvVerify — replaces the magic 'ROCM_ENV_VERIFIED' string with a typed verdict."""
from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext

_VERIFY_SCRIPT = """
set -e
python - <<'PY'
import torch, sys, json
ok = torch.cuda.is_available()
info = {
    "torch_version": torch.__version__,
    "cuda_available": ok,
    "device_count": torch.cuda.device_count() if ok else 0,
    "device_name": (torch.cuda.get_device_name(0) if ok else None),
}
print("ENV_VERIFY_JSON:" + json.dumps(info))
sys.exit(0 if ok else 1)
PY
"""


class EnvVerifyInput(BaseModel):
    custom_command: str | None = None  # override the default torch.cuda check


class EnvVerifyOutput(BaseModel):
    verdict: Literal["ok", "no_gpu", "import_error", "unknown"]
    detail: str
    raw_stdout: str
    raw_stderr: str


class EnvVerify(BaseTool[EnvVerifyInput, EnvVerifyOutput]):
    name: ClassVar[str] = "EnvVerify"
    description: ClassVar[str] = (
        "Verify the container's ROCm/CUDA environment is functional by running "
        "torch.cuda.is_available() (or a custom command). Returns a typed verdict."
    )
    input_model: ClassVar[type[BaseModel]] = EnvVerifyInput
    max_result_size_chars: ClassVar[int] = 8_000

    def is_concurrency_safe(self, parsed: EnvVerifyInput) -> bool:
        return False  # spawns a python process; serialize

    def is_read_only(self, parsed: EnvVerifyInput) -> bool:
        return True

    async def call(
        self, parsed: EnvVerifyInput, ctx: ToolUseContext
    ) -> ToolResult[EnvVerifyOutput]:
        if ctx.sandbox is None:
            return ToolResult(
                data=EnvVerifyOutput(
                    verdict="unknown", detail="no sandbox", raw_stdout="", raw_stderr=""
                ),
                text="no sandbox attached",
                is_error=True,
            )
        cmd = parsed.custom_command or _VERIFY_SCRIPT
        res = await ctx.sandbox.exec(cmd, timeout_s=180.0)
        verdict: Literal["ok", "no_gpu", "import_error", "unknown"] = "unknown"
        detail = ""
        if res.exit_code == 0:
            verdict = "ok"
            detail = "torch.cuda.is_available() returned True"
            # mark gate state so EnvVerify-hooks can proceed
            gate = getattr(ctx, "gate_state", None)
            if gate is not None and hasattr(gate, "mark_gpu_check"):
                gate.mark_gpu_check()
                gate.mark_env_verified()
        elif "ModuleNotFoundError" in res.stderr or "ImportError" in res.stderr:
            verdict = "import_error"
            detail = "torch not installed or broken"
        else:
            verdict = "no_gpu"
            detail = "torch installed but cuda.is_available() is False"
        return ToolResult(
            data=EnvVerifyOutput(
                verdict=verdict, detail=detail, raw_stdout=res.stdout, raw_stderr=res.stderr
            ),
            text=(
                f"EnvVerify -> verdict={verdict}\n{detail}\n\n"
                f"stdout:\n{res.stdout[-2000:]}\n\n"
                f"stderr:\n{res.stderr[-2000:]}"
            ),
            is_error=verdict != "ok",
        )
