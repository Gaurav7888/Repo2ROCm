from repo2rocm.tools.skills.invoke_skill import InvokeSkill
from repo2rocm.tools.base import register_tool


def register_skill_tools() -> None:
    register_tool(InvokeSkill)


__all__ = ["InvokeSkill", "register_skill_tools"]
