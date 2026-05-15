"""Dream Phase 5: Extract - discover automatable patterns and create tools."""

from __future__ import annotations

import logging
from typing import Any

from laimiu.procedural.engine import ProceduralEngine

logger = logging.getLogger("laimiu.dream.extract")


class ProceduralExtractor:
    """Phase 5: Extract - convert repeated patterns into executable tools.

    This is the key innovation of v3: the Dream engine doesn't just
    organize text memories, it creates new capabilities.
    """

    def __init__(self, procedural_engine: ProceduralEngine):
        self.engine = procedural_engine

    async def extract(self) -> dict[str, Any]:
        """Run the extraction pipeline.

        Returns summary of tools created.
        """
        actions: dict[str, Any] = {
            "patterns_analyzed": 0,
            "tools_generated": [],
            "errors": [],
        }

        try:
            # Find extractable patterns (using strength-based filtering)
            from laimiu.procedural.extractor import PatternExtractor

            patterns = self.engine.extractor.find_extractable_patterns()
            actions["patterns_analyzed"] = len(patterns)

            if not patterns:
                logger.info("No patterns to extract")
                return actions

            logger.info(
                f"Found {len(patterns)} extractable patterns "
                f"(strength >= {self.engine.config.procedural.extract_strength})"
            )

            # Generate tools from patterns
            new_tools = await self.engine.run_extraction()
            actions["tools_generated"] = new_tools

            # Clear extracted patterns from tracker
            for pattern in patterns:
                key = f"{pattern.tool_name}:{pattern.args_signature}"
                self.engine.tracker.clear_pattern(key)

        except Exception as e:
            logger.error(f"Procedural extraction failed: {e}")
            actions["errors"].append(str(e))

        return actions
