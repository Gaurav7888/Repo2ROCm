"""LLM API client. Multi-provider, streaming, prompt-cache aware.

Public surface:
  * `ModelClient` protocol
  * `AnthropicClient` (direct API + AMD gateway flavor)
  * `OpenAIClient` (optional dep)
  * `MockClient` (testing)
  * `stream_model(...)` — the unified streaming entry point yielding `StreamChunk` events

`StreamChunk` is what the agent loop consumes. Each chunk is one of:
  - `ChunkText`     — partial text from the assistant
  - `ChunkToolUse`  — a fully-formed tool_use block (input json complete)
  - `ChunkThinking` — partial thinking text
  - `ChunkUsage`    — token usage update (sent at stream end)
  - `ChunkError`    — error event (may be withheld by the loop)
  - `ChunkDone`     — terminal: final AssistantMessage assembled
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from repo2rocm.core.messages import (
    AssistantMessage,
    Message,
    SystemPrompt,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolUseBlock,
)
from repo2rocm.observability.logging import get_logger
from repo2rocm.observability.metrics import METRICS
from repo2rocm.observability.tracing import span

log = get_logger(__name__)


# ── Stream chunk types ────────────────────────────────────────────────────────


@dataclass
class ChunkText:
    text: str
    block_index: int


@dataclass
class ChunkThinking:
    thinking: str
    block_index: int


@dataclass
class ChunkToolUse:
    tool_use: ToolUseBlock
    block_index: int


@dataclass
class ChunkUsage:
    usage: TokenUsage


@dataclass
class ChunkError:
    error_class: str
    message: str
    recoverable: bool = False
    raw: Any | None = None


@dataclass
class ChunkDone:
    assistant_message: AssistantMessage
    stop_reason: str | None = None


StreamChunk = ChunkText | ChunkThinking | ChunkToolUse | ChunkUsage | ChunkError | ChunkDone


# ── Tool spec (what we send the model) ────────────────────────────────────────


@dataclass
class ToolSpec:
    """Provider-agnostic tool spec, derived from BaseTool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    cache_control: dict[str, Any] | None = None


# ── Client protocol ──────────────────────────────────────────────────────────


class FallbackTriggered(Exception):
    def __init__(self, fallback_model: str):
        self.fallback_model = fallback_model
        super().__init__(f"falling back to {fallback_model}")


class ModelClient(Protocol):
    name: str
    model: str

    async def stream(
        self,
        *,
        messages: list[Message],
        system: SystemPrompt,
        tools: list[ToolSpec],
        max_tokens: int,
        temperature: float = 0.0,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[StreamChunk]: ...


# ── Anthropic implementation ──────────────────────────────────────────────────


@dataclass
class AnthropicClient:
    model: str
    api_key: str = ""
    base_url: str = "https://api.anthropic.com"
    api_version: str = "2023-06-01"
    timeout: float = 600.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    name: str = "anthropic"

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def stream(
        self,
        *,
        messages: list[Message],
        system: SystemPrompt,
        tools: list[ToolSpec],
        max_tokens: int,
        temperature: float = 0.0,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[StreamChunk]:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "system": [_system_block_to_api(b) for b in system.blocks],
            "messages": _messages_to_api(messages),
            "tools": [_tool_spec_to_api(t) for t in tools],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
            "accept": "text/event-stream",
            **self.extra_headers,
        }
        url = f"{self.base_url}/v1/messages"

        with span("llm.stream", model=self.model, provider="anthropic"):
            with METRICS.time_llm(self.model):
                async for chunk in self._stream_sse(url, headers, payload, stop_event):
                    yield chunk

    async def _stream_sse(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        stop_event: asyncio.Event | None,
    ) -> AsyncGenerator[StreamChunk, None]:
        client = await self._get_client()
        assembled_blocks: list[Any] = []
        partial_tool_json: dict[int, str] = {}
        partial_tool_meta: dict[int, dict[str, Any]] = {}
        partial_text: dict[int, str] = {}
        partial_thinking: dict[int, str] = {}
        usage = TokenUsage()
        stop_reason: str | None = None

        try:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    METRICS.llm_errors.labels(model=self.model, error_class=str(resp.status_code)).inc()
                    yield ChunkError(
                        error_class=f"http_{resp.status_code}",
                        message=body.decode("utf-8", errors="replace")[:2000],
                        recoverable=resp.status_code in (529,),
                    )
                    return
                async for line in resp.aiter_lines():
                    if stop_event is not None and stop_event.is_set():
                        return
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        ev = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    et = ev.get("type")
                    if et == "content_block_start":
                        idx = ev["index"]
                        block = ev["content_block"]
                        partial_tool_meta[idx] = {}
                        if block["type"] == "tool_use":
                            partial_tool_meta[idx] = {
                                "id": block["id"],
                                "name": block["name"],
                            }
                            partial_tool_json[idx] = ""
                        elif block["type"] == "text":
                            partial_text[idx] = ""
                        elif block["type"] == "thinking":
                            partial_thinking[idx] = ""
                    elif et == "content_block_delta":
                        idx = ev["index"]
                        d = ev["delta"]
                        dt = d.get("type")
                        if dt == "text_delta":
                            clean = _strip_chat_template_tokens(d["text"])
                            partial_text[idx] = partial_text.get(idx, "") + clean
                            if clean:
                                yield ChunkText(text=clean, block_index=idx)
                        elif dt == "input_json_delta":
                            partial_tool_json[idx] += d.get("partial_json", "")
                        elif dt == "thinking_delta":
                            partial_thinking[idx] = partial_thinking.get(idx, "") + d["thinking"]
                            yield ChunkThinking(thinking=d["thinking"], block_index=idx)
                    elif et == "content_block_stop":
                        idx = ev["index"]
                        if idx in partial_tool_json:
                            try:
                                tool_input = json.loads(partial_tool_json[idx] or "{}")
                            except json.JSONDecodeError:
                                tool_input = {"_raw": partial_tool_json[idx]}
                            tool_meta = partial_tool_meta.get(idx, {})
                            tu = ToolUseBlock(
                                id=tool_meta.get("id", f"toolu_{idx}"),
                                name=_sanitize_tool_name(tool_meta.get("name", "unknown")),
                                input=tool_input,
                            )
                            assembled_blocks.append(tu)
                            yield ChunkToolUse(tool_use=tu, block_index=idx)
                        elif idx in partial_text:
                            assembled_blocks.append(TextBlock(text=partial_text[idx]))
                        elif idx in partial_thinking:
                            assembled_blocks.append(ThinkingBlock(thinking=partial_thinking[idx]))
                    elif et == "message_delta":
                        d = ev.get("delta", {})
                        if "stop_reason" in d:
                            stop_reason = d["stop_reason"]
                        u = ev.get("usage", {})
                        usage = TokenUsage(
                            input_tokens=u.get("input_tokens", usage.input_tokens),
                            output_tokens=u.get("output_tokens", usage.output_tokens),
                            cache_creation_input_tokens=u.get(
                                "cache_creation_input_tokens", usage.cache_creation_input_tokens
                            ),
                            cache_read_input_tokens=u.get(
                                "cache_read_input_tokens", usage.cache_read_input_tokens
                            ),
                        )
                    elif et == "message_stop":
                        # Final usage may also appear here
                        pass
                    elif et == "error":
                        err = ev.get("error", {})
                        METRICS.llm_errors.labels(
                            model=self.model, error_class=err.get("type", "unknown")
                        ).inc()
                        yield ChunkError(
                            error_class=err.get("type", "unknown"),
                            message=err.get("message", ""),
                            recoverable=False,
                            raw=err,
                        )
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            METRICS.llm_errors.labels(model=self.model, error_class=type(e).__name__).inc()
            yield ChunkError(
                error_class=type(e).__name__,
                message=str(e),
                recoverable=True,
            )
            return

        # Emit usage + done
        if usage.total > 0:
            METRICS.llm_tokens.labels(model=self.model, kind="input").observe(usage.input_tokens)
            METRICS.llm_tokens.labels(model=self.model, kind="output").observe(usage.output_tokens)
            if usage.cache_read_input_tokens:
                METRICS.llm_tokens.labels(model=self.model, kind="cache_read").observe(
                    usage.cache_read_input_tokens
                )
            if usage.cache_creation_input_tokens:
                METRICS.llm_tokens.labels(model=self.model, kind="cache_creation").observe(
                    usage.cache_creation_input_tokens
                )
            METRICS.cache_hit_ratio.labels(model=self.model).set(usage.cache_hit_ratio())
            yield ChunkUsage(usage=usage)

        assistant = AssistantMessage(
            content=assembled_blocks,
            model=self.model,
            stop_reason=stop_reason,
            usage=usage,
        )
        yield ChunkDone(assistant_message=assistant, stop_reason=stop_reason)


# ── AMD Gateway client (Claude-compatible) ────────────────────────────────────


@dataclass
class AMDGatewayClient:
    """AMD LLM API Gateway (On-Prem) — pure OpenAI chat-completions wrapper.

    Reference invocation (the AMD-supplied example):

        openai.OpenAI(
            base_url="https://llm-api.amd.com/OnPrem",
            api_key="dummy",
            default_headers={
                "Ocp-Apim-Subscription-Key": "<key>",
                "user": os.getlogin(),
            },
        ).chat.completions.create(model="GPT-oss-20B", ...)

    So:
      * URL: POST {base_url}/chat/completions   (model goes in BODY, OpenAI-style)
      * Headers: `Ocp-Apim-Subscription-Key` + `user` (required by the gateway)
      * Body: OpenAI {model, messages: [{role, content}], temperature, tools, ...}
      * Response: JSON. Gateway may return EITHER OpenAI (`choices[0].message.*`)
        OR Anthropic-shaped (`content[]`) — we accept both.
      * Tool calling: OpenAI `tools` / `tool_calls` format.

    The agent loop is unchanged; we just emit the same StreamChunk sequence
    (ChunkText / ChunkToolUse / ChunkUsage / ChunkDone).
    """

    model: str
    api_key: str = ""
    base_url: str = "https://llm-api.amd.com/OnPrem"
    user: str = ""  # the AMD gateway requires a `user` header
    timeout: float = 600.0
    name: str = "amd_gateway"
    # `protocol` is auto-derived from base_url in __post_init__ unless caller sets it.
    #   "openai"    → /OnPrem style: model in BODY, OpenAI-shape tools/messages
    #   "anthropic" → /claude3 style: model in PATH, Anthropic-shape tools/system
    protocol: str = ""

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("AMD_LLM_API_KEY", "")
        if not self.user:
            self.user = (
                os.environ.get("REPO2ROCM_AMD_GATEWAY_USER")
                or os.environ.get("USER")
                or os.environ.get("USERNAME")
                or "repo2rocm"
            )
        # auto-detect protocol from URL unless caller forced one
        if not self.protocol:
            self.protocol = (
                "anthropic" if "/claude3" in self.base_url else "openai"
            )
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def stream(
        self,
        *,
        messages: list[Message],
        system: SystemPrompt,
        tools: list[ToolSpec],
        max_tokens: int,
        temperature: float = 0.0,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[StreamChunk]:
        # The gateway caps output around 16K tokens regardless of model.
        capped_max_tokens = min(max_tokens, 16_000)

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "user": self.user,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.protocol == "anthropic":
            # /claude3/<model>/chat/completions — model in PATH, Anthropic body shape
            sys_text = "\n\n".join(b.text for b in system.blocks).strip()
            body: dict[str, Any] = {
                "messages": _messages_to_api(messages),  # Anthropic style
                "max_tokens": capped_max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            if sys_text:
                body["system"] = sys_text
            if tools:
                body["tools"] = [_tool_spec_to_api(t) for t in tools]
            url = f"{self.base_url.rstrip('/')}/{self.model}/chat/completions"
        else:
            # /OnPrem/chat/completions — model in BODY, OpenAI shape
            openai_messages: list[dict[str, Any]] = []
            sys_text = "\n\n".join(b.text for b in system.blocks).strip()
            if sys_text:
                openai_messages.append({"role": "system", "content": sys_text})
            openai_messages.extend(_messages_to_openai(messages))

            body = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature,
                "stream": False,
                "max_completion_tokens": capped_max_tokens,
            }
            if tools:
                body["tools"] = [_tool_spec_to_openai(t) for t in tools]
                body["tool_choice"] = "auto"
            url = f"{self.base_url.rstrip('/')}/chat/completions"

        with span(
            "llm.stream",
            model=self.model,
            provider="amd_gateway",
            protocol=self.protocol,
        ):
            with METRICS.time_llm(self.model):
                async for chunk in self._post_and_parse(url, headers, body, stop_event):
                    yield chunk

    async def _post_and_parse(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        stop_event: asyncio.Event | None,
    ) -> AsyncGenerator[StreamChunk, None]:
        client = await self._get_client()
        try:
            resp = await client.post(url, headers=headers, json=body)
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            METRICS.llm_errors.labels(model=self.model, error_class=type(exc).__name__).inc()
            yield ChunkError(
                error_class=type(exc).__name__, message=str(exc), recoverable=True
            )
            return

        if resp.status_code >= 400:
            METRICS.llm_errors.labels(
                model=self.model, error_class=str(resp.status_code)
            ).inc()
            yield ChunkError(
                error_class=f"http_{resp.status_code}",
                message=resp.text[:2000],
                recoverable=resp.status_code in (429, 502, 503, 504, 529),
            )
            return

        try:
            data = resp.json()
        except (ValueError, json.JSONDecodeError) as exc:
            yield ChunkError(
                error_class="invalid_json", message=str(exc), recoverable=False
            )
            return

        # ── Parse — accept either OpenAI or Anthropic shape ──
        text_parts: list[str] = []
        tool_uses: list[ToolUseBlock] = []
        stop_reason: str | None = None
        usage = TokenUsage()

        # NOTE: small models (notably gpt-oss-20b) leak Harmony chat-template tokens
        # into `function.name` — e.g. "Agent<|channel|>commentary" instead of "Agent".
        # Sanitize at parse time so the tool registry lookup succeeds.

        # Anthropic-style: {"content": [{"type":"text","text":...}, {"type":"tool_use",...}]}
        if isinstance(data.get("content"), list):
            for block in data["content"]:
                btype = block.get("type")
                if btype == "text":
                    t = _strip_chat_template_tokens(block.get("text", ""))
                    text_parts.append(t)
                elif btype == "tool_use":
                    tool_uses.append(
                        ToolUseBlock(
                            id=block.get("id", f"toolu_{len(tool_uses)}"),
                            name=_sanitize_tool_name(block.get("name", "unknown")),
                            input=block.get("input", {}),
                        )
                    )
            stop_reason = data.get("stop_reason")
            u = data.get("usage") or {}
            usage = TokenUsage(
                input_tokens=int(u.get("input_tokens", 0)),
                output_tokens=int(u.get("output_tokens", 0)),
            )
        # OpenAI-style: {"choices":[{"message":{"content":"...","tool_calls":[...]}}]}
        elif isinstance(data.get("choices"), list) and data["choices"]:
            msg = data["choices"][0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content:
                text_parts.append(_strip_chat_template_tokens(content))
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        parsed_args = {"_raw": raw_args}
                else:
                    parsed_args = raw_args
                tool_uses.append(
                    ToolUseBlock(
                        id=tc.get("id", f"toolu_{len(tool_uses)}"),
                        name=_sanitize_tool_name(fn.get("name", "unknown")),
                        input=parsed_args,
                    )
                )
            stop_reason = data["choices"][0].get("finish_reason") or msg.get("stop_reason")
            # OpenAI calls them prompt_tokens / completion_tokens
            u = data.get("usage") or {}
            usage = TokenUsage(
                input_tokens=int(u.get("prompt_tokens", u.get("input_tokens", 0))),
                output_tokens=int(u.get("completion_tokens", u.get("output_tokens", 0))),
            )
        else:
            yield ChunkError(
                error_class="unexpected_response_shape",
                message=str(data)[:1000],
                recoverable=False,
            )
            return

        # ── Synthesize the same StreamChunk sequence as the streaming impl ──
        combined_text = "".join(text_parts)
        if combined_text:
            yield ChunkText(text=combined_text, block_index=0)
        for i, tu in enumerate(tool_uses):
            yield ChunkToolUse(tool_use=tu, block_index=i + 1)
        if usage.total:
            METRICS.llm_tokens.labels(model=self.model, kind="input").observe(usage.input_tokens)
            METRICS.llm_tokens.labels(model=self.model, kind="output").observe(usage.output_tokens)
            yield ChunkUsage(usage=usage)

        # Build the canonical AssistantMessage for the loop.
        content_blocks: list[Any] = []
        if combined_text:
            content_blocks.append(TextBlock(text=combined_text))
        content_blocks.extend(tool_uses)
        assistant = AssistantMessage(
            content=content_blocks,
            model=self.model,
            stop_reason=stop_reason,
            usage=usage,
        )
        yield ChunkDone(assistant_message=assistant, stop_reason=stop_reason)


# ── Mock client for tests ─────────────────────────────────────────────────────


@dataclass
class MockClient:
    """Deterministic mock: returns a scripted sequence of chunks."""

    name: str = "mock"
    model: str = "mock-model"
    scripted_responses: list[list[StreamChunk]] = field(default_factory=list)
    _call_index: int = 0

    async def stream(
        self,
        *,
        messages: list[Message],
        system: SystemPrompt,
        tools: list[ToolSpec],
        max_tokens: int,
        temperature: float = 0.0,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[StreamChunk]:
        if self._call_index >= len(self.scripted_responses):
            # default empty completion
            am = AssistantMessage(content=[TextBlock(text="(end)")])
            yield ChunkDone(assistant_message=am)
            return
        chunks = self.scripted_responses[self._call_index]
        self._call_index += 1
        for ch in chunks:
            yield ch


# ── Unified entry point ───────────────────────────────────────────────────────


async def stream_model(
    *,
    client: ModelClient,
    messages: list[Message],
    system: SystemPrompt,
    tools: list[ToolSpec],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    stop_event: asyncio.Event | None = None,
) -> AsyncIterator[StreamChunk]:
    """Stream from `client` with one retry on transient errors."""
    last_chunks: list[StreamChunk] = []
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(2),
            wait=wait_random_exponential(min=1, max=10),
            retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
            reraise=True,
        ):
            with attempt:
                last_chunks = []
                async for ch in client.stream(
                    messages=messages,
                    system=system,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_event=stop_event,
                ):
                    last_chunks.append(ch)
    except RetryError:
        pass
    for ch in last_chunks:
        yield ch


# ── Adapters: pydantic models <-> Anthropic JSON ──────────────────────────────


def _system_block_to_api(block: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "text", "text": block.text}
    if block.cache_control:
        out["cache_control"] = block.cache_control
    return out


def _messages_to_api(messages: list[Message]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        if hasattr(m, "kind"):  # SystemMessage → translate to user
            out.append({"role": "user", "content": [{"type": "text", "text": m.content}]})
            continue
        if isinstance(m.content, str):
            out.append({"role": m.role, "content": m.content})
        else:
            out.append({"role": m.role, "content": [_block_to_api(b) for b in m.content]})
    return out


def _block_to_api(block: Any) -> dict[str, Any]:
    d = block.model_dump(exclude_none=True)
    # pydantic uses snake_case which matches Anthropic JSON for these blocks
    return d


_HARMONY_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_HARMONY_NAME_TAIL_RE = re.compile(r"<\|[^|]*\|>.*$")


def _sanitize_tool_name(raw: str) -> str:
    """Strip model-template artifacts (e.g. Harmony `<|channel|>commentary`) from tool names.

    Small open-source models (gpt-oss-20b in particular) leak chat-template channel
    markers into the OpenAI `function.name` field, producing names like
    `Agent<|channel|>commentary`. We strip everything from the first `<|...|>` onward
    so the registry lookup succeeds.
    """
    if not raw:
        return "unknown"
    name = _HARMONY_NAME_TAIL_RE.sub("", raw).strip()
    # also strip any whitespace or non-identifier trailing junk
    # The OpenAI tools spec requires name ~ /^[A-Za-z0-9_-]{1,64}$/
    name = re.sub(r"[^A-Za-z0-9_\-]+", "", name)
    return name or "unknown"


def _strip_chat_template_tokens(text: str | None) -> str:
    """Strip embedded Harmony chat-template tokens from text body.

    GPT-OSS-20B (and similar Harmony-format models) sometimes emit literal
    `<|end|>`, `<|start|>assistant`, `<|channel|>commentary`, `<|message|>`
    inside their content. If we echo that text back in the next turn's messages,
    the gateway tokenizer rejects it with HTTP 400. Strip the markers so the
    conversation history stays well-formed.
    """
    if not text:
        return text or ""
    return _HARMONY_TOKEN_RE.sub("", text)


def _tool_spec_to_api(spec: ToolSpec) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
    }
    if spec.cache_control:
        out["cache_control"] = spec.cache_control
    return out


# ── OpenAI-shape adapters (used by AMDGatewayClient) ──────────────────────────


def _messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert our internal Message list to OpenAI-style chat messages.

    Lossy where required:
      * tool_use blocks become `assistant` messages with `tool_calls`
      * tool_result blocks become `tool` messages
      * thinking blocks are dropped (the gateway doesn't accept them)
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        # Internal SystemMessage — flatten to a system message
        if hasattr(m, "kind"):
            out.append({"role": "system", "content": str(m.content)})
            continue

        role = m.role
        if isinstance(m.content, str):
            out.append({"role": role, "content": m.content})
            continue

        # m.content is a list of blocks
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []  # → emitted as separate tool messages

        for b in m.content:
            btype = getattr(b, "type", None)
            if btype == "text":
                text_parts.append(_strip_chat_template_tokens(b.text))
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": b.id,
                        "type": "function",
                        "function": {
                            "name": b.name,
                            "arguments": json.dumps(b.input or {}),
                        },
                    }
                )
            elif btype == "tool_result":
                content = b.content if isinstance(b.content, str) else json.dumps(b.content)
                if not content:
                    content = "[internal-empty-tool-result]"
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": b.tool_use_id,
                        "content": content,
                    }
                )
            elif btype == "thinking":
                # not representable; drop
                continue

        if role == "assistant":
            asst_msg: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                asst_msg["content"] = "".join(text_parts)
            else:
                asst_msg["content"] = None  # OpenAI permits null content when tool_calls present
            if tool_calls:
                asst_msg["tool_calls"] = tool_calls
            out.append(asst_msg)
            # tool_results should NOT appear in an assistant message; if they did, emit them next
            for tr in tool_results:
                out.append(tr)
        else:
            # user role — text + any tool_result blocks become separate tool messages
            if text_parts:
                out.append({"role": "user", "content": "".join(text_parts)})
            for tr in tool_results:
                out.append(tr)
            if not text_parts and not tool_results and not tool_calls:
                # Provider-compatible clients reject empty user messages.
                out.append(
                    {
                        "role": role,
                        "content": "[internal-empty-user-message]",
                    }
                )

    return out


def _tool_spec_to_openai(spec: ToolSpec) -> dict[str, Any]:
    """Convert a ToolSpec to OpenAI function-tool format."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description[:1024],
            "parameters": spec.input_schema,
        },
    }
