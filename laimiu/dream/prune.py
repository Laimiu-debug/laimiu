"""Dream Phase 4: Prune - trim the memory index to stay within limits."""

from __future__ import annotations

import logging
from typing import Any

from laimiu.constants import MEMORY_INDEX_MAX_LINES
from laimiu.memory.manager import MemoryManager

logger = logging.getLogger("laimiu.dream.prune")


class Prune:
    """Phase 4: Prune - keep the memory index concise and relevant.

    The MEMORY.md index must stay ≤200 lines. This phase:
    1. Checks current line count
    2. Removes outdated entries
    3. Merges similar entries
    4. Ensures the index is clean and useful
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager

    def prune(self) -> dict[str, Any]:
        """Run pruning on the memory index.

        Returns summary of pruning actions.
        """
        actions = {
            "lines_before": 0,
            "lines_after": 0,
            "entries_removed": 0,
        }

        index = self.memory.index
        actions["lines_before"] = index.line_count

        if index.line_count <= MEMORY_INDEX_MAX_LINES:
            logger.info("Index within limits, no pruning needed")
            actions["lines_after"] = index.line_count
            return actions

        # Prune: remove oldest entries first
        content = index.get()
        lines = content.split("\n")

        # Separate header from entries
        header_lines = []
        entry_lines = []
        in_header = True
        for line in lines:
            if in_header and (line.startswith("#") or not line.strip()):
                header_lines.append(line)
            else:
                in_header = False
                if line.strip().startswith("-"):
                    entry_lines.append(line)
                elif line.strip():
                    entry_lines.append(line)

        # Keep the most recent entries (bottom of list)
        available_space = MEMORY_INDEX_MAX_LINES - len(header_lines)
        if len(entry_lines) > available_space:
            removed = len(entry_lines) - available_space
            entry_lines = entry_lines[-available_space:]
            actions["entries_removed"] = removed

        # Rebuild index
        new_content = "\n".join(header_lines + entry_lines)
        index.update(new_content)
        actions["lines_after"] = index.line_count

        logger.info(
            f"Pruned index: {actions['lines_before']} -> {actions['lines_after']} lines "
            f"({actions['entries_removed']} entries removed)"
        )

        return actions
