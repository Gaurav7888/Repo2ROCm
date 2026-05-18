"""Message types: the entire wire format that flows through the agent loop.

These mirror the Anthropic Messages API shape but are provider-agnostic.
Every adapter (Anthropic, OpenAI, AMD gateway) converts to/from these types.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# ── Content blocks ────────────────────────────────────────────────────────────


class TextBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["text"] = "text"
    text: str
    cache_control: dict[str, Any] | None = None


class ThinkingBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None  # model-bound; strip on fallback


class ToolUseBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[dict[str, Any]]
    is_error: bool = False


class ImageBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["image"] = "image"
    source: dict[str, Any]


ContentBlock = Annotated[
    Union[TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock, ImageBlock],
    Field(discriminator="type"),
]


# ── Top-level messages ────────────────────────────────────────────────────────


class UserMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["user"] = "user"
    content: str | list[ContentBlock]


class AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]
    model: str | None = None
    stop_reason: str | None = None
    usage: TokenUsage | None = None

    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


class SystemMessage(BaseModel):
    """Internal-only: boundary markers, hook outputs, recall attachments."""

    model_config = ConfigDict(extra="ignore")
    role: Literal["system"] = "system"
    content: str
    kind: str = "info"  # "boundary" | "hook" | "recall" | "info"


Message = Annotated[
    Union[UserMessage, AssistantMessage, SystemMessage],
    Field(discriminator="role"),
]


# ── System prompt ─────────────────────────────────────────────────────────────


class SystemPromptBlock(BaseModel):
    """One block of the system prompt; we use multiple for cache stability."""

    model_config = ConfigDict(extra="ignore")
    text: str
    cache_control: dict[str, Any] | None = None  # set on the last STABLE block


class SystemPrompt(BaseModel):
    model_config = ConfigDict(extra="ignore")
    blocks: list[SystemPromptBlock]

    @classmethod
    def from_text(cls, text: str, *, cacheable: bool = True) -> SystemPrompt:
        ctrl = {"type": "ephemeral"} if cacheable else None
        return cls(blocks=[SystemPromptBlock(text=text, cache_control=ctrl)])

    def total_chars(self) -> int:
        return sum(len(b.text) for b in self.blocks)


# ── Token usage ───────────────────────────────────────────────────────────────


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def cache_hit_ratio(self) -> float:
        denom = self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens
        if denom == 0:
            return 0.0
        return self.cache_read_input_tokens / denom

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
        )


# resolve forward ref on AssistantMessage
AssistantMessage.model_rebuild()
