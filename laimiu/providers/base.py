"""Base provider types for Laimiu."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderProfile:
    """Profile for a single LLM provider."""

    name: str
    base_url: str
    model: str
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    supports_tools: bool = True
    supports_streaming: bool = True

    @property
    def api_mode(self) -> str:
        """Detect API compatibility mode."""
        url = self.base_url.lower()
        if "openai" in url or "deepseek" in url or "bigmodel" in url or "/v1" in url:
            return "openai"
        if "localhost:11434" in url or "ollama" in url:
            return "ollama"
        return "openai"  # default to OpenAI-compatible


@dataclass
class Message:
    """A chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None  # DeepSeek thinking mode


@dataclass
class ToolCall:
    """A parsed tool call from LLM response."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class StreamChunk:
    """A chunk of streamed response."""

    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    reasoning_content: str = ""  # DeepSeek thinking mode


@dataclass
class LLMResponse:
    """Complete response from LLM."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
