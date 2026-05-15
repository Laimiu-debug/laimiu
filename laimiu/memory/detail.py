"""Tier 2: Detailed topic notes (Markdown, loaded on demand)."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from laimiu.constants import NOTES_DIR

logger = logging.getLogger("laimiu.memory.detail")


class DetailStore:
    """Manages Tier 2 topic notes.

    Notes are Markdown files in ~/.laimiu/memory/notes/.
    Each note focuses on a single topic and is indexed by ChromaDB.
    """

    def __init__(self, notes_dir: Path | None = None):
        self.notes_dir = notes_dir or NOTES_DIR
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def save_note(self, topic: str, content: str, metadata: dict | None = None) -> str:
        """Save a topic note. Returns the note ID."""
        # Generate a safe filename from the topic
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic)
        safe_name = safe_name[:60]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        note_id = f"{safe_name}_{timestamp}"
        note_path = self.notes_dir / f"{note_id}.md"

        # Add metadata header
        header = f"---\ntopic: {topic}\ncreated: {datetime.now().isoformat()}\n"
        if metadata:
            for k, v in metadata.items():
                header += f"{k}: {v}\n"
        header += "---\n\n"

        note_path.write_text(header + content, encoding="utf-8")
        logger.debug(f"Saved note: {note_id}")
        return note_id

    def read_note(self, note_id: str) -> str | None:
        """Read a note by ID."""
        note_path = self.notes_dir / f"{note_id}.md"
        if not note_path.exists():
            return None
        return note_path.read_text(encoding="utf-8")

    def list_notes(self) -> list[dict]:
        """List all notes with basic metadata."""
        notes = []
        for md_file in sorted(self.notes_dir.glob("*.md"), reverse=True):
            content = md_file.read_text(encoding="utf-8")
            topic = md_file.stem
            # Parse YAML-like header
            lines = content.split("\n")
            if lines[0] == "---":
                for line in lines[1:]:
                    if line == "---":
                        break
                    if line.startswith("topic:"):
                        topic = line.split(":", 1)[1].strip()

            notes.append({
                "id": md_file.stem,
                "topic": topic,
                "size": len(content),
                "modified": md_file.stat().st_mtime,
            })
        return notes

    def delete_note(self, note_id: str) -> bool:
        """Delete a note by ID."""
        note_path = self.notes_dir / f"{note_id}.md"
        if note_path.exists():
            note_path.unlink()
            return True
        return False

    def get_all_content(self) -> list[tuple[str, str]]:
        """Get all notes as (note_id, content) tuples for vectorization."""
        result = []
        for md_file in sorted(self.notes_dir.glob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            result.append((md_file.stem, content))
        return result
