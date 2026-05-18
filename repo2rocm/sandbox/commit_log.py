"""Commit DAG: each turn produces a child commit; rollbacks create branches.

The old stack-based rollback prevented two migrators from operating speculatively on
disjoint file sets. With a DAG, a worker can `branch_from(commit_id)` and run an
alternate plan without disturbing the trunk.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class CommitNode:
    id: str
    parent_id: str | None
    image: str           # docker image id at this commit
    timestamp: float
    label: str           # human-readable: e.g. "after-pip-install"
    children: list[str] = field(default_factory=list)


@dataclass
class CommitLog:
    nodes: dict[str, CommitNode] = field(default_factory=dict)
    head: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, *, id: str, parent_id: str | None, image: str, label: str = "") -> CommitNode:
        with self._lock:
            node = CommitNode(
                id=id,
                parent_id=parent_id,
                image=image,
                timestamp=time.time(),
                label=label,
            )
            self.nodes[id] = node
            if parent_id and parent_id in self.nodes:
                self.nodes[parent_id].children.append(id)
            self.head = id
            return node

    def set_head(self, commit_id: str) -> None:
        with self._lock:
            if commit_id not in self.nodes:
                raise KeyError(commit_id)
            self.head = commit_id

    def trunk(self) -> list[CommitNode]:
        """Linear chain from root → head along parent links."""
        chain: list[CommitNode] = []
        cur = self.head
        while cur is not None:
            node = self.nodes.get(cur)
            if node is None:
                break
            chain.append(node)
            cur = node.parent_id
        return list(reversed(chain))

    def to_dict(self) -> dict:
        return {
            "head": self.head,
            "nodes": {k: v.__dict__ for k, v in self.nodes.items()},
        }
