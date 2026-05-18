"""Agent registry — track running sub-agents for SendMessage / TaskStop / auto-resume."""
from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from repo2rocm.agents.definition import AgentDefinition
from repo2rocm.observability.metrics import METRICS


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class AgentTaskState:
    id: str
    name: str
    type: str  # e.g. "local_agent"
    agent_def: AgentDefinition
    status: TaskStatus = TaskStatus.PENDING
    output_file: Path | None = None
    started_at: float = 0.0
    ended_at: float | None = None
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    pending_messages: list[str] = field(default_factory=list)
    transcript_path: Path | None = None
    final_text: str = ""

    def mark_running(self) -> None:
        self.status = TaskStatus.RUNNING
        METRICS.subagent_active.labels(agent_type=self.agent_def.name).inc()

    def mark_terminal(self, status: TaskStatus, final_text: str = "") -> None:
        self.status = status
        self.final_text = final_text
        METRICS.subagent_active.labels(agent_type=self.agent_def.name).dec()


@dataclass
class AgentRegistry:
    tasks: dict[str, AgentTaskState] = field(default_factory=dict)
    name_index: dict[str, str] = field(default_factory=dict)  # human name → task id
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def register(self, *, agent_def: AgentDefinition, name: str | None = None) -> AgentTaskState:
        with self._lock:
            tid = f"a{uuid.uuid4().hex[:8]}"
            ts = AgentTaskState(
                id=tid,
                name=name or agent_def.name,
                type="local_agent",
                agent_def=agent_def,
            )
            self.tasks[tid] = ts
            if name:
                self.name_index[name] = tid
            return ts

    def resolve(self, name_or_id: str) -> AgentTaskState | None:
        with self._lock:
            if name_or_id in self.tasks:
                return self.tasks[name_or_id]
            tid = self.name_index.get(name_or_id)
            return self.tasks.get(tid) if tid else None

    def kill(self, task_id: str) -> bool:
        with self._lock:
            ts = self.tasks.get(task_id)
            if ts is None:
                return False
            ts.abort_event.set()
            if ts.task is not None:
                ts.task.cancel()
            ts.mark_terminal(TaskStatus.KILLED)
            return True


_REGISTRY = AgentRegistry()


def get_agent_registry() -> AgentRegistry:
    return _REGISTRY
