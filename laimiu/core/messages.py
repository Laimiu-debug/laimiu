"""Structured output messages for CLI rendering layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OutputMessage:
    """A typed output message from the agent to the CLI renderer.

    Types:
      - "content": normal assistant reply text
      - "thinking": reasoning/thinking content (DeepSeek-R1 etc.)
      - "tool_call": tool invocation started {name, args}
      - "tool_result": tool execution result {name, elapsed, success}
      - "error": error occurred {error}
      - "system": system-level message (session start, dream, etc.)
    """

    type: str
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "brain"
    task_id: str = ""

    @classmethod
    def content_chunk(cls, text: str) -> OutputMessage:
        return cls(type="content", content=text)

    @classmethod
    def thinking(cls, text: str) -> OutputMessage:
        return cls(type="thinking", content=text)

    @classmethod
    def tool_call_start(cls, name: str, args: dict | None = None) -> OutputMessage:
        return cls(type="tool_call", content=name, metadata={"args": args or {}, "status": "running"})

    @classmethod
    def tool_call_end(cls, name: str, elapsed_ms: float, success: bool) -> OutputMessage:
        return cls(
            type="tool_result",
            content=name,
            metadata={"elapsed_ms": elapsed_ms, "success": success, "status": "done"},
        )

    @classmethod
    def error(cls, message: str, detail: str = "") -> OutputMessage:
        return cls(type="error", content=message, metadata={"detail": detail})

    @classmethod
    def system(cls, message: str) -> OutputMessage:
        return cls(type="system", content=message)
