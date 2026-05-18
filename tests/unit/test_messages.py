"""Pydantic message types round-trip cleanly."""
from __future__ import annotations

from repo2rocm.core.messages import (
    AssistantMessage,
    SystemPrompt,
    TextBlock,
    TokenUsage,
    ToolUseBlock,
    UserMessage,
)


def test_assistant_message_text_concatenation():
    am = AssistantMessage(
        content=[TextBlock(text="hello "), TextBlock(text="world")],
    )
    assert am.text() == "hello world"
    assert am.tool_uses() == []


def test_assistant_message_collects_tool_uses():
    tu = ToolUseBlock(id="t1", name="Read", input={"file_path": "a.py"})
    am = AssistantMessage(content=[TextBlock(text="reading"), tu])
    assert am.tool_uses() == [tu]


def test_user_message_supports_str_and_blocks():
    um1 = UserMessage(content="plain text")
    um2 = UserMessage(content=[TextBlock(text="block")])
    assert um1.role == "user"
    assert um2.content[0].text == "block"


def test_token_usage_addition_and_cache_hit_ratio():
    a = TokenUsage(input_tokens=100, output_tokens=20, cache_read_input_tokens=80)
    b = TokenUsage(input_tokens=50, output_tokens=10)
    s = a + b
    assert s.input_tokens == 150
    assert s.output_tokens == 30
    assert s.cache_read_input_tokens == 80
    assert 0.0 <= a.cache_hit_ratio() <= 1.0


def test_system_prompt_from_text_marks_cache_control():
    sp = SystemPrompt.from_text("hello")
    assert sp.blocks[0].cache_control == {"type": "ephemeral"}
    assert sp.total_chars() == 5
