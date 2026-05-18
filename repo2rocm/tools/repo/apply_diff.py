"""ApplyDiff — accept a SEARCH/REPLACE diff block, apply to a target file.

Diff format (compatible with the old Repo2ROCm code_edit.py):

    <<<<<<< SEARCH
    old text
    =======
    new text
    >>>>>>> REPLACE
"""
from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext
from repo2rocm.tools.repo.edit import Edit, EditInput

_SR_PATTERN = re.compile(
    r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE",
    re.DOTALL,
)


class ApplyDiffInput(BaseModel):
    file_path: str
    diff: str  # one or more SEARCH/REPLACE hunks


class ApplyDiffOutput(BaseModel):
    file_path: str
    hunks_applied: int


class ApplyDiff(BaseTool[ApplyDiffInput, ApplyDiffOutput]):
    name: ClassVar[str] = "ApplyDiff"
    description: ClassVar[str] = (
        "Apply one or more SEARCH/REPLACE hunks to a file. Each hunk must exactly match."
    )
    input_model: ClassVar[type[BaseModel]] = ApplyDiffInput
    max_result_size_chars: ClassVar[int] = 4_000

    def is_concurrency_safe(self, parsed: ApplyDiffInput) -> bool:
        return False

    def is_read_only(self, parsed: ApplyDiffInput) -> bool:
        return False

    async def call(
        self, parsed: ApplyDiffInput, ctx: ToolUseContext
    ) -> ToolResult[ApplyDiffOutput]:
        hunks = list(_SR_PATTERN.finditer(parsed.diff))
        if not hunks:
            return ToolResult(
                data=ApplyDiffOutput(file_path=parsed.file_path, hunks_applied=0),
                text=(
                    "No SEARCH/REPLACE blocks found in diff. Expected:\n"
                    "<<<<<<< SEARCH\n...old...\n=======\n...new...\n>>>>>>> REPLACE"
                ),
                is_error=True,
            )

        edit = Edit()
        applied = 0
        last_text = ""
        for m in hunks:
            old, new = m.group(1), m.group(2)
            result = await edit.call(
                EditInput(file_path=parsed.file_path, old_string=old, new_string=new),
                ctx,
            )
            last_text = result.text
            if result.is_error:
                return ToolResult(
                    data=ApplyDiffOutput(file_path=parsed.file_path, hunks_applied=applied),
                    text=f"Hunk {applied + 1} failed: {result.text}",
                    is_error=True,
                )
            applied += 1
        return ToolResult(
            data=ApplyDiffOutput(file_path=parsed.file_path, hunks_applied=applied),
            text=f"Applied {applied} hunk(s) to {parsed.file_path}",
        )
