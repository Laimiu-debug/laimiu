"""Reflection module - post-tool-call self-evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from laimiu.tools.base import ToolResult

logger = logging.getLogger("laimiu.core.reflection")


@dataclass
class ReflectionResult:
    """Result of reflecting on a tool execution."""

    confidence: float  # 0.0-1.0
    should_retry: bool = False
    learned: str | None = None
    alternative_approach: str | None = None
    summary: str = ""


class Reflection:
    """Evaluates tool execution results and decides on follow-up actions.

    The reflection loop drives learning:
    - Tool failure -> analyze cause -> try alternative
    - Tool success -> note if a better approach exists
    - Repeated patterns -> feed into procedural memory tracker
    """

    def evaluate(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
        original_intent: str,
    ) -> ReflectionResult:
        """Reflect on a tool execution.

        Args:
            tool_name: Name of the tool that was called.
            args: Arguments passed to the tool.
            result: The tool's execution result.
            original_intent: What the user originally asked for.

        Returns:
            ReflectionResult with confidence, retry decision, and learnings.
        """
        if result.success:
            return self._evaluate_success(tool_name, args, result, original_intent)
        else:
            return self._evaluate_failure(tool_name, args, result, original_intent)

    def _evaluate_success(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
        original_intent: str,
    ) -> ReflectionResult:
        """Evaluate a successful tool call."""
        output_len = len(result.output)

        # Heuristic confidence based on output
        confidence = 0.8
        if output_len > 100:
            confidence = 0.9  # Substantial output suggests good result
        if output_len == 0:
            confidence = 0.6  # Empty output might be incomplete

        return ReflectionResult(
            confidence=confidence,
            should_retry=False,
            summary=f"Tool '{tool_name}' succeeded ({output_len} chars output)",
        )

    def _evaluate_failure(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
        original_intent: str,
    ) -> ReflectionResult:
        """Evaluate a failed tool call and suggest alternatives."""
        error = result.error.lower()

        # Classify the failure
        alternative = None
        learned = None

        if "not found" in error or "no such file" in error:
            alternative = "Try searching for the file first with search_files"
            learned = "File path may be incorrect, search before accessing"

        elif "timeout" in error:
            alternative = "Try with a longer timeout or simpler command"
            learned = "Operation may be too slow, increase timeout"

        elif "permission" in error or "denied" in error:
            alternative = "Check permissions or try a different approach"
            learned = f"Permission issue with {tool_name}"

        elif "unknown tool" in error:
            alternative = "Use one of the available tools instead"
            learned = "Tool name was incorrect"

        else:
            alternative = "Try a different approach or tool"
            learned = f"Tool '{tool_name}' failed: {result.error[:100]}"

        return ReflectionResult(
            confidence=0.2,
            should_retry=True,
            learned=learned,
            alternative_approach=alternative,
            summary=f"Tool '{tool_name}' failed: {result.error[:100]}",
        )
