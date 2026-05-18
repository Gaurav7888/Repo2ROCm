from repo2rocm.tools.external.pypi import PyPIVersions
from repo2rocm.tools.external.docker_hub import DockerHubTags
from repo2rocm.tools.external.web_search import WebSearch
from repo2rocm.tools.external.fetch import Fetch
from repo2rocm.tools.base import register_tool


def register_external_tools() -> None:
    for cls in (PyPIVersions, DockerHubTags, WebSearch, Fetch):
        register_tool(cls)


__all__ = [
    "PyPIVersions",
    "DockerHubTags",
    "WebSearch",
    "Fetch",
    "register_external_tools",
]
