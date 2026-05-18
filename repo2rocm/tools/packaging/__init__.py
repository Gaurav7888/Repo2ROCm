from repo2rocm.tools.packaging.waiting_list import (
    WaitingListAdd,
    WaitingListAddFile,
    WaitingListShow,
    WaitingListClear,
    WaitingList,
)
from repo2rocm.tools.packaging.conflict_list import (
    ConflictList,
    ConflictListShow,
    ConflictListSolve,
    ConflictListClear,
)
from repo2rocm.tools.packaging.download import Download
from repo2rocm.tools.base import register_tool


def register_packaging_tools() -> None:
    for cls in (
        WaitingListAdd,
        WaitingListAddFile,
        WaitingListShow,
        WaitingListClear,
        ConflictListShow,
        ConflictListSolve,
        ConflictListClear,
        Download,
    ):
        register_tool(cls)


__all__ = [
    "WaitingList",
    "ConflictList",
    "WaitingListAdd",
    "WaitingListAddFile",
    "WaitingListShow",
    "WaitingListClear",
    "ConflictListShow",
    "ConflictListSolve",
    "ConflictListClear",
    "Download",
    "register_packaging_tools",
]
