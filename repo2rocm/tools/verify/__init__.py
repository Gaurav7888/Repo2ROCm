from repo2rocm.tools.verify.env_verify import EnvVerify
from repo2rocm.tools.verify.paper_verify import PaperVerify
from repo2rocm.tools.base import register_tool


def register_verify_tools() -> None:
    for cls in (EnvVerify, PaperVerify):
        register_tool(cls)


__all__ = ["EnvVerify", "PaperVerify", "register_verify_tools"]
