"""Procedural engine - orchestrates the procedural memory lifecycle."""

from __future__ import annotations

import logging
from typing import Any

from laimiu.config.settings import LaimiuConfig
from laimiu.procedural.extractor import PatternExtractor
from laimiu.procedural.tool_writer import ToolWriter
from laimiu.procedural.tracker import PatternTracker
from laimiu.providers.router import ProviderRouter
from laimiu.tools.registry import ToolRegistry

logger = logging.getLogger("laimiu.procedural.engine")


class ProceduralEngine:
    """Manages the full lifecycle of procedural memory (Layer 2).

    Lifecycle:
    1. Track: PatternTracker records tool call patterns
    2. Extract: PatternExtractor identifies automatable patterns
    3. Generate: LLM writes a Python tool script
    4. Validate: ToolValidator checks for safety
    5. Register: ToolWriter saves and registers the new tool
    """

    def __init__(
        self,
        config: LaimiuConfig,
        tracker: PatternTracker,
        registry: ToolRegistry,
        router: ProviderRouter,
    ):
        self.config = config
        self.tracker = tracker
        self.registry = registry
        self.router = router
        self.extractor = PatternExtractor(tracker, config.procedural.extract_strength)
        self.writer = ToolWriter(registry)

    async def run_extraction(self) -> list[str]:
        """Run the full extraction pipeline.

        Returns list of newly created tool names.
        """
        if not self.config.procedural.enabled:
            logger.info("Procedural memory disabled")
            return []

        # Find extractable patterns
        patterns = self.extractor.find_extractable_patterns()
        if not patterns:
            logger.info("No extractable patterns found")
            return []

        logger.info(f"Found {len(patterns)} extractable patterns")

        new_tools = []
        for pattern in patterns:
            try:
                tool_name = await self._generate_tool(pattern)
                if tool_name:
                    new_tools.append(tool_name)
                    logger.info(f"Generated tool: {tool_name}")
            except Exception as e:
                logger.error(f"Failed to generate tool from pattern: {e}")

        return new_tools

    async def _generate_tool(self, pattern: Any) -> str | None:
        """Generate a tool script from a pattern using LLM."""
        # Generate prompt for LLM
        prompt = self.extractor.generate_tool_prompt(pattern)

        # Call LLM to generate the script
        from laimiu.providers.base import Message

        messages = [
            Message(
                role="system",
                content="You are a Python code generator. Write clean, safe, well-documented code.",
            ),
            Message(role="user", content=prompt),
        ]

        try:
            response = await self.router.chat_complete(messages, task="cheap")
            script_content = response.content

            if not script_content:
                logger.error("LLM returned empty response for tool generation")
                return None

            # Write and register the tool
            tool_name = await self.writer.write_and_register(script_content)
            return tool_name

        except Exception as e:
            logger.error(f"Tool generation failed: {e}")
            return None

    def get_stats(self) -> dict[str, Any]:
        """Get procedural memory statistics."""
        patterns = self.tracker.get_all_patterns()
        extractable = self.tracker.get_extractable_patterns()

        # Breakdown by level
        level_counts: dict[str, int] = {}
        for p in patterns:
            level_counts[p.level] = level_counts.get(p.level, 0) + 1

        return {
            "total_patterns": len(patterns),
            "extractable_patterns": len(extractable),
            "generated_tools": len(self.writer.list_generated_tools()),
            "level_breakdown": level_counts,
        }
