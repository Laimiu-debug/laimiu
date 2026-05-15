"""Tier 1: MEMORY.md index - always loaded, ≤200 lines."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from laimiu.constants import MEMORY_INDEX_FILE, MEMORY_INDEX_MAX_LINES
from laimiu.utils.io import atomic_write

logger = logging.getLogger("laimiu.memory.index")


class MemoryIndex:
    """Manages the MEMORY.md index file (Tier 1 memory).

    The index is a concise summary of all stored memories.
    It should never exceed MEMORY_INDEX_MAX_LINES lines.
    """

    def __init__(self, path: Path | None = None):
        self.path = path or MEMORY_INDEX_FILE
        self._content: str = ""
        self._load()

    def _load(self) -> None:
        """Load the index from disk."""
        if self.path.exists():
            self._content = self.path.read_text(encoding="utf-8")
        else:
            self._content = "# Laimiu Memory Index\n# Auto-managed. Do not edit manually.\n\n"

    def get(self) -> str:
        """Get the current index content."""
        return self._content

    def reload(self) -> None:
        """Reload from disk."""
        self._load()

    def update(self, new_content: str) -> None:
        """Update the index with new content, enforcing line limit."""
        lines = new_content.strip().split("\n")
        if len(lines) > MEMORY_INDEX_MAX_LINES:
            logger.warning(
                f"Memory index has {len(lines)} lines, pruning to {MEMORY_INDEX_MAX_LINES}"
            )
            # Keep header and most recent entries
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith("#") or not line.strip():
                    header_end = i + 1
                else:
                    break
            header = lines[:header_end]
            body = lines[header_end:]
            lines = header + body[-(MEMORY_INDEX_MAX_LINES - len(header)):]

        self._content = "\n".join(lines) + "\n"
        self._save()

    def add_entry(self, category: str, summary: str, note_path: str = "") -> None:
        """Add a new entry to the index."""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        entry = f"- [{timestamp}] {category}: {summary}"
        if note_path:
            entry += f" → {note_path}"
        self._content = self._content.rstrip("\n") + "\n" + entry + "\n"
        self._save()

    def remove_entry(self, keyword: str) -> bool:
        """Remove entries containing a keyword."""
        lines = self._content.split("\n")
        new_lines = [l for l in lines if keyword.lower() not in l.lower()]
        if len(new_lines) == len(lines):
            return False
        self._content = "\n".join(new_lines) + "\n"
        self._save()
        return True

    def _save(self) -> None:
        """Save the index to disk atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(self.path, self._content)

    @property
    def line_count(self) -> int:
        return len(self._content.strip().split("\n"))
