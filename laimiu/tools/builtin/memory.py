"""Memory recall tool - retrieves memories from ChromaDB."""

from __future__ import annotations

from typing import Any

from laimiu.tools.base import BaseTool, ToolResult


class MemoryRecallTool(BaseTool):
    name = "memory_recall"
    description = "Search your memory for relevant information. Use this when you need to recall past conversations, user preferences, or learned knowledge."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in memory",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, memory_manager=None):
        self._memory = memory_manager

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)

        if not query:
            return ToolResult(success=False, error="No query provided")

        if self._memory is None:
            return ToolResult(success=False, error="Memory system not initialized")

        try:
            results = self._memory.search(query, max_results=max_results)
            return ToolResult(success=True, output=results)
        except Exception as e:
            return ToolResult(success=False, error=f"Memory search failed: {e}")
