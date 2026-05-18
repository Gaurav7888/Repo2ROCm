"""Regression: AnthropicClient.stream() must work end-to-end without AttributeError.

The original bug: `core/api.py` used `async with span(...)` where `span` is a SYNC
contextmanager. The unit suite missed it because every other test uses MockClient.
This test stubs httpx's transport so we exercise the real streaming code path
without making a network call.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from repo2rocm.core.api import (
    AMDGatewayClient,
    AnthropicClient,
    ChunkDone,
    ChunkError,
    ChunkText,
    ChunkToolUse,
    ChunkUsage,
    StreamChunk,
    ToolSpec,
    _messages_to_openai,
    _tool_spec_to_openai,
)
from repo2rocm.core.messages import (
    AssistantMessage,
    SystemPrompt,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)


# A canned Anthropic Messages-API SSE response: one text block + one tool_use + usage.
_SSE_BODY = b"""\
event: message_start
data: {"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"model":"claude-3-5-sonnet","usage":{"input_tokens":12,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_1","name":"Read","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"file_path\\":\\"a.py\\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"input_tokens":12,"output_tokens":7}}

event: message_stop
data: {"type":"message_stop"}

"""


def _stub_handler(request: httpx.Request) -> httpx.Response:
    """httpx MockTransport handler — returns the canned SSE stream."""
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_SSE_BODY,
    )


def _err_handler(status: int):
    def h(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=b'{"error":{"type":"overloaded","message":"slow down"}}')

    return h


async def _collect(it: AsyncIterator[StreamChunk]) -> list[StreamChunk]:
    out: list[StreamChunk] = []
    async for c in it:
        out.append(c)
    return out


@pytest.mark.asyncio
async def test_anthropic_stream_parses_text_and_tool_use(monkeypatch):
    """The exact code path that crashed in production. Must complete without AttributeError."""
    client = AnthropicClient(model="claude-3-5-sonnet", api_key="fake")
    # Inject our stubbed transport so no real network call happens.
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(_stub_handler))
    try:
        chunks = await _collect(
            client.stream(
                messages=[UserMessage(content="hi")],
                system=SystemPrompt.from_text("you are a test"),
                tools=[],
                max_tokens=128,
            )
        )
    finally:
        await client.aclose()

    kinds = [type(c).__name__ for c in chunks]
    # We expect text deltas, a tool_use, usage, and a final done — no exceptions.
    assert "ChunkText" in kinds
    assert "ChunkToolUse" in kinds
    assert "ChunkUsage" in kinds
    assert "ChunkDone" in kinds

    text = "".join(c.text for c in chunks if isinstance(c, ChunkText))
    assert text == "hello world"

    tools = [c for c in chunks if isinstance(c, ChunkToolUse)]
    assert len(tools) == 1
    assert tools[0].tool_use.name == "Read"
    assert tools[0].tool_use.input == {"file_path": "a.py"}

    usage = next(c for c in chunks if isinstance(c, ChunkUsage))
    assert usage.usage.input_tokens == 12
    assert usage.usage.output_tokens == 7

    done = next(c for c in chunks if isinstance(c, ChunkDone))
    assert done.assistant_message.text() == "hello world"
    assert len(done.assistant_message.tool_uses()) == 1
    assert done.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_anthropic_stream_handles_http_error():
    """Non-2xx response must surface as a ChunkError, not raise."""
    client = AnthropicClient(model="claude-3-5-sonnet", api_key="fake")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(_err_handler(529)))
    try:
        chunks = await _collect(
            client.stream(
                messages=[UserMessage(content="hi")],
                system=SystemPrompt.from_text("test"),
                tools=[],
                max_tokens=128,
            )
        )
    finally:
        await client.aclose()
    from repo2rocm.core.api import ChunkError

    errs = [c for c in chunks if isinstance(c, ChunkError)]
    assert len(errs) == 1
    assert errs[0].error_class == "http_529"
    assert errs[0].recoverable is True  # 529 = overloaded → retryable


# ── AMD Gateway tests ─────────────────────────────────────────────────────────


_OPENAI_RESPONSE_WITH_TOOL_CALL = {
    "id": "chatcmpl-fake",
    "object": "chat.completion",
    "model": "claude-sonnet-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Let me read that file.",
                "tool_calls": [
                    {
                        "id": "toolu_abc",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": '{"file_path":"a.py"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59},
}


_ANTHROPIC_SHAPED_RESPONSE = {
    "id": "msg_fake",
    "model": "claude-sonnet-4",
    "content": [
        {"type": "text", "text": "hello from anthropic shape"},
        {"type": "tool_use", "id": "toolu_xyz", "name": "Glob", "input": {"pattern": "*.py"}},
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 30, "output_tokens": 8},
}


def _amd_handler(payload_capture: dict, response_body: dict):
    """Make an httpx MockTransport handler that captures the request and returns `response_body`."""

    def h(request: httpx.Request) -> httpx.Response:
        payload_capture["url"] = str(request.url)
        payload_capture["method"] = request.method
        payload_capture["headers"] = dict(request.headers)
        import json as _json

        payload_capture["body"] = _json.loads(request.content.decode())
        return httpx.Response(200, json=response_body)

    return h


@pytest.mark.asyncio
async def test_amd_gateway_request_shape_and_openai_response():
    """End-to-end: AMDGatewayClient builds an OpenAI-shape POST, parses OpenAI response."""
    captured: dict = {}
    client = AMDGatewayClient(model="claude-sonnet-4", api_key="fake-amd-key")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_amd_handler(captured, _OPENAI_RESPONSE_WITH_TOOL_CALL))
    )
    try:
        chunks = await _collect(
            client.stream(
                messages=[UserMessage(content="please read a.py")],
                system=SystemPrompt.from_text("you are a test agent"),
                tools=[
                    ToolSpec(
                        name="Read",
                        description="read a file",
                        input_schema={
                            "type": "object",
                            "properties": {"file_path": {"type": "string"}},
                            "required": ["file_path"],
                        },
                    )
                ],
                max_tokens=8192,
            )
        )
    finally:
        await client.aclose()

    # The URL must include the model in the path, not in the body.
    assert captured["url"].endswith("/claude3/claude-sonnet-4/chat/completions"), captured["url"]
    # AMD-specific auth header
    assert captured["headers"].get("ocp-apim-subscription-key") == "fake-amd-key"
    # OpenAI body shape
    body = captured["body"]
    assert body["stream"] is False
    assert "messages" in body and isinstance(body["messages"], list)
    # system became a leading {"role":"system",...}
    assert body["messages"][0]["role"] == "system"
    assert "you are a test agent" in body["messages"][0]["content"]
    # tools converted to OpenAI function-tool shape
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "Read"

    # Parsed back to our internal chunks
    text = "".join(c.text for c in chunks if isinstance(c, ChunkText))
    assert text == "Let me read that file."
    tools = [c for c in chunks if isinstance(c, ChunkToolUse)]
    assert len(tools) == 1
    assert tools[0].tool_use.name == "Read"
    assert tools[0].tool_use.input == {"file_path": "a.py"}
    usage = next(c for c in chunks if isinstance(c, ChunkUsage))
    assert usage.usage.input_tokens == 42
    assert usage.usage.output_tokens == 17
    done = next(c for c in chunks if isinstance(c, ChunkDone))
    assert done.stop_reason == "tool_calls"


@pytest.mark.asyncio
async def test_amd_gateway_accepts_anthropic_shape_response():
    """Some gateway models return Anthropic-style {content:[{type:...}]}."""
    captured: dict = {}
    client = AMDGatewayClient(model="claude-sonnet-4", api_key="fake")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_amd_handler(captured, _ANTHROPIC_SHAPED_RESPONSE))
    )
    try:
        chunks = await _collect(
            client.stream(
                messages=[UserMessage(content="hi")],
                system=SystemPrompt.from_text("test"),
                tools=[],
                max_tokens=2048,
            )
        )
    finally:
        await client.aclose()

    text = "".join(c.text for c in chunks if isinstance(c, ChunkText))
    assert text == "hello from anthropic shape"
    tools = [c for c in chunks if isinstance(c, ChunkToolUse)]
    assert len(tools) == 1
    assert tools[0].tool_use.name == "Glob"


@pytest.mark.asyncio
async def test_amd_gateway_404_yields_chunkerror():
    """A 404 (the original failure) must surface as ChunkError, not raise."""

    def h(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text='{"error":"Resource Not Found"}')

    client = AMDGatewayClient(model="bad-model", api_key="fake")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(h))
    try:
        chunks = await _collect(
            client.stream(
                messages=[UserMessage(content="hi")],
                system=SystemPrompt.from_text("test"),
                tools=[],
                max_tokens=512,
            )
        )
    finally:
        await client.aclose()

    errs = [c for c in chunks if isinstance(c, ChunkError)]
    assert len(errs) == 1
    assert errs[0].error_class == "http_404"
    assert errs[0].recoverable is False  # 404 is not retryable


# ── OpenAI-shape converter unit tests ─────────────────────────────────────────


def test_messages_to_openai_round_trip_assistant_with_tool_use():
    asst = AssistantMessage(
        content=[
            TextBlock(text="reading"),
            ToolUseBlock(id="tu1", name="Read", input={"file_path": "a.py"}),
        ]
    )
    out = _messages_to_openai([asst])
    assert len(out) == 1
    assert out[0]["role"] == "assistant"
    assert out[0]["content"] == "reading"
    assert out[0]["tool_calls"][0]["function"]["name"] == "Read"
    import json as _json

    assert _json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {"file_path": "a.py"}


def test_messages_to_openai_emits_tool_message_for_results():
    user_with_results = UserMessage(
        content=[ToolResultBlock(tool_use_id="tu1", content="file contents here")]
    )
    out = _messages_to_openai([user_with_results])
    # The tool_result became a separate role=tool message
    assert len(out) == 1
    assert out[0]["role"] == "tool"
    assert out[0]["tool_call_id"] == "tu1"
    assert out[0]["content"] == "file contents here"


def test_tool_spec_to_openai_shape():
    spec = ToolSpec(
        name="Read",
        description="read a file",
        input_schema={"type": "object", "properties": {"file_path": {"type": "string"}}},
    )
    out = _tool_spec_to_openai(spec)
    assert out["type"] == "function"
    assert out["function"]["name"] == "Read"
    assert out["function"]["parameters"]["type"] == "object"
