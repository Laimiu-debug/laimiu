"""Unified OpenAI-compatible LLM client with lazy import."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from laimiu.providers.base import (
    LLMResponse,
    Message,
    ProviderProfile,
    StreamChunk,
    ToolCall,
)

logger = logging.getLogger("laimiu.providers")


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert our Message objects to OpenAI API format."""
    result = []
    for msg in messages:
        d: dict[str, Any] = {"role": msg.role}
        if msg.content:
            d["content"] = msg.content
        if msg.tool_calls:
            d["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        if msg.name:
            d["name"] = msg.name
        result.append(d)
    return result


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert tool definitions to OpenAI function-calling format."""
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return openai_tools


def _parse_tool_calls(raw_calls: list[Any]) -> list[ToolCall]:
    """Parse OpenAI tool call objects into our ToolCall type."""
    calls = []
    for tc in raw_calls:
        fn = tc.function
        args = {}
        if fn.arguments:
            try:
                args = json.loads(fn.arguments)
            except json.JSONDecodeError:
                args = {"raw": fn.arguments}
        calls.append(ToolCall(id=tc.id, name=fn.name, arguments=args))
    return calls


class OpenAICompatProvider:
    """Provider using OpenAI-compatible API (works with DeepSeek, GLM, Ollama, etc.)."""

    def __init__(self, profile: ProviderProfile):
        self.profile = profile
        self._client = None

    def _get_client(self):
        """Lazy-init the OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            kwargs: dict[str, Any] = {
                "base_url": self.profile.base_url,
            }
            if self.profile.api_key:
                kwargs["api_key"] = self.profile.api_key
            else:
                kwargs["api_key"] = "not-needed"  # For Ollama

            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Send messages and stream response chunks."""
        client = self._get_client()
        api_messages = _to_openai_messages(messages)

        api_kwargs: dict[str, Any] = {
            "model": self.profile.model,
            "messages": api_messages,
            "temperature": self.profile.temperature,
            "max_tokens": self.profile.max_tokens,
            "stream": stream,
        }
        if tools:
            api_kwargs["tools"] = _to_openai_tools(tools)
        api_kwargs.update(kwargs)

        response = await client.chat.completions.create(**api_kwargs)

        if stream:
            # Collect tool call deltas across chunks
            tool_call_buffers: dict[int, dict[str, str]] = {}

            async for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                content = delta.content or ""

                # Accumulate tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_buffers:
                            tool_call_buffers[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        buf = tool_call_buffers[idx]
                        if tc_delta.id:
                            buf["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                buf["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                buf["arguments"] += tc_delta.function.arguments

                # When finish_reason signals completion, yield accumulated tool calls
                if choice.finish_reason in ("tool_calls", "stop"):
                    # Build complete tool call data from buffers
                    complete_tc = []
                    for idx in sorted(tool_call_buffers.keys()):
                        buf = tool_call_buffers[idx]
                        if buf["name"]:  # Only yield if we have a function name
                            complete_tc.append({
                                "index": idx,
                                "id": buf["id"],
                                "function": {
                                    "name": buf["name"],
                                    "arguments": buf["arguments"],
                                },
                            })
                    yield StreamChunk(
                        content=content,
                        tool_calls=complete_tc if complete_tc else None,
                        finish_reason=choice.finish_reason,
                    )
                    # Reset buffers for potential multi-turn tool calls
                    tool_call_buffers.clear()
                elif content:
                    # Normal text chunk
                    yield StreamChunk(
                        content=content,
                        finish_reason=None,
                    )
        else:
            # Non-streaming
            choice = response.choices[0]
            content = choice.message.content or ""
            tool_calls = []
            if choice.message.tool_calls:
                tool_calls = _parse_tool_calls(choice.message.tool_calls)

            # Yield as single chunk
            result = LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason or "",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            )
            yield StreamChunk(content=content, finish_reason=result.finish_reason)

    async def chat_complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Non-streaming chat - returns full response."""
        client = self._get_client()
        api_messages = _to_openai_messages(messages)

        api_kwargs: dict[str, Any] = {
            "model": self.profile.model,
            "messages": api_messages,
            "temperature": self.profile.temperature,
            "max_tokens": self.profile.max_tokens,
            "stream": False,
        }
        if tools:
            api_kwargs["tools"] = _to_openai_tools(tools)
        api_kwargs.update(kwargs)

        response = await client.chat.completions.create(**api_kwargs)

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            tool_calls = _parse_tool_calls(choice.message.tool_calls)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
        )
