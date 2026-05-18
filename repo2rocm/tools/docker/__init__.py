from repo2rocm.tools.docker.exec import DockerExec
from repo2rocm.tools.docker.commit import DockerCommit, DockerRollback
from repo2rocm.tools.docker.change_image import ChangeBaseImage, ChangePythonVersion
from repo2rocm.tools.base import register_tool


def register_docker_tools() -> None:
    for cls in (DockerExec, DockerCommit, DockerRollback, ChangeBaseImage, ChangePythonVersion):
        register_tool(cls)


__all__ = [
    "DockerExec",
    "DockerCommit",
    "DockerRollback",
    "ChangeBaseImage",
    "ChangePythonVersion",
    "register_docker_tools",
]
