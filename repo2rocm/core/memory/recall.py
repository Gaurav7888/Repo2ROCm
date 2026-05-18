"""RecallSelector — cheap LLM side-query that picks ≤5 memory files for the current turn."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from repo2rocm.core.api import ModelClient
from repo2rocm.core.memory.staleness import staleness_warning
from repo2rocm.core.memory.store import MemoryFile, MemoryStore
from repo2rocm.core.messages import SystemPrompt, UserMessage
from repo2rocm.observability.tracing import span


@dataclass
class RecallSelector:
    client: ModelClient
    store: MemoryStore
    max_select: int = 5

    async def select_for(self, query: str) -> list[MemoryFile]:
        files = self.store.list_files()
        if not files:
            return []

        with span("memory.recall", candidates=len(files)):
            prompt = (
                "You are selecting which previously-saved memory files are relevant "
                "to the user's current task. Return JSON only:\n"
                '{"selected": ["file1.md", "file2.md", ...]}\n'
                "Rules: prefer FEWER selections; skip if uncertain. Max 5.\n\n"
                f"User task:\n{query}\n\nAvailable memories:\n{self.store.manifest_for_recall()}"
            )
            try:
                # use a one-shot non-streaming call would be cleaner; here we synthesize
                # via the streaming API and concatenate text.
                from repo2rocm.core.api import ChunkDone, stream_model

                msg = UserMessage(content=prompt)
                text = ""
                async for chunk in stream_model(
                    client=self.client,
                    messages=[msg],
                    system=SystemPrompt.from_text(
                        "You are a precise JSON-only selector.", cacheable=False
                    ),
                    tools=[],
                    max_tokens=512,
                ):
                    if isinstance(chunk, ChunkDone):
                        text = chunk.assistant_message.text()
            except Exception:
                return []

        try:
            data = json.loads(text)
            selected_names = data.get("selected", [])
        except (json.JSONDecodeError, ValueError):
            return []

        known = {f.path.name: f for f in files}
        return [known[n] for n in selected_names if n in known][: self.max_select]

    def render(self, files: list[MemoryFile]) -> str:
        """Format selected memories with staleness warnings for injection into the system prompt."""
        if not files:
            return ""
        chunks: list[str] = ["# Recalled Memories"]
        for mf in files:
            body = self.store.load_body(mf)
            warn = staleness_warning(mf.age_days())
            chunks.append(f"## {mf.name}  ({mf.type})\n{warn}\n{body}".rstrip())
        return "\n\n".join(chunks)
