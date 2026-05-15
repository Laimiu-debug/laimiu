"""Tool executor - dispatches tool calls with safety checks and result reporting."""

from __future__ import annotations

import logging
from typing import Any

from laimiu.providers.base import ToolCall
from laimiu.tools.base import ToolResult
from laimiu.tools.registry import ToolRegistry

logger = logging.getLogger("laimiu.core.tool_executor")


class ToolExecutor:
    """Executes tool calls from the LLM with safety and tracking."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._call_log: list[dict[str, Any]] = []

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call.

        Args:
            tool_call: The parsed tool call from LLM response.

        Returns:
            ToolResult with success/failure and output.
        """
        logger.debug(f"Executing tool: {tool_call.name}({tool_call.arguments})")

        result = await self.registry.execute(tool_call.name, tool_call.arguments)

        # Log the call
        self._call_log.append({
            "tool": tool_call.name,
            "args": tool_call.arguments,
            "success": result.success,
            "output_len": len(result.output),
            "error": result.error[:200] if result.error else "",
        })

        if result.success:
            logger.debug(f"Tool '{tool_call.name}' succeeded")
        else:
            logger.warning(f"Tool '{tool_call.name}' failed: {result.error[:100]}")

        return result

    async def execute_batch(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple independent tool calls.

        For MVP, we execute sequentially. Future: parallel for independent calls.
        """
        results = []
        for tc in tool_calls:
            result = await self.execute(tc)
            results.append(result)
        return results

    def get_call_log(self) -> list[dict[str, Any]]:
        """Get the log of all tool calls in this session."""
        return self._call_log.copy()

    def clear_log(self) -> None:
        """Clear the call log."""
        self._call_log.clear()
