"""Base tool types for Laimiu."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    output: str = ""
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Serialize to text for LLM consumption."""
        if self.success:
            return self.output if self.output else "Done."
        return f"Error: {self.error}"


@dataclass
class ToolDefinition:
    """Definition of a tool for LLM function calling."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Any = None  # Callable

    def to_openai_spec(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class BaseTool:
    """Base class for all Laimiu tools."""

    # Subclasses must define these
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name and cls.name != BaseTool.name:
            # Auto-register when subclass is defined
            _PENDING_TOOLS.append(cls)

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments."""
        raise NotImplementedError

    def get_definition(self) -> ToolDefinition:
        """Get tool definition for LLM."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            handler=self.execute,
        )


# Temporary holding list for auto-discovered tools
_PENDING_TOOLS: list[type[BaseTool]] = []
