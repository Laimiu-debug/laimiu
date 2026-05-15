"""Dream Phase 5: Extract - discover automatable patterns and create tools."""

from __future__ import annotations

import logging
from typing import Any

from laimiu.procedural.engine import ProceduralEngine

logger = logging.getLogger("laimiu.dream.extract")


class ProceduralExtractor:
    """Phase 5: Extract - convert repeated patterns into executable tools."""

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
            # Get existing tool names for logging
            existing = {t.name for t in self.engine.registry.list_tools()}

            # Run extraction (engine handles dedup internally)
            new_tools = await self.engine.run_extraction()
            actions["tools_generated"] = new_tools
            actions["patterns_analyzed"] = len(
                self.engine.extractor.find_extractable_patterns(existing_tools=existing)
            )

            if not new_tools:
                logger.info("No new tools generated")
                return actions

            logger.info(f"Generated {len(new_tools)} new tools: {new_tools}")

            # Clear ALL extractable patterns from tracker after extraction
            # (whether they were successfully extracted or not)
            all_extractable = self.engine.tracker.get_extractable_patterns()
            for pattern in all_extractable:
                key = f"{pattern.tool_name}:{pattern.args_signature}"
                self.engine.tracker.clear_pattern(key)

        except Exception as e:
            logger.error(f"Procedural extraction failed: {e}")
            actions["errors"].append(str(e))

        return actions
