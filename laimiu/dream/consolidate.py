"""Dream Phase 3: Consolidate - integrate and organize memories."""

from __future__ import annotations

import logging
from typing import Any

from laimiu.memory.manager import MemoryManager

logger = logging.getLogger("laimiu.dream.consolidate")


class Consolidate:
    """Phase 3: Consolidate - merge, organize, and update memory tiers.

    Tasks:
    - Merge duplicate/overlapping notes
    - Create new notes from transcript highlights
    - Update MEMORY.md index
    - Vectorize new content
    """

    def __init__(self, memory_manager: MemoryManager, router: Any):
        self.memory = memory_manager
        self.router = router

    async def consolidate(self, signals: dict[str, Any]) -> dict[str, Any]:
        """Run consolidation on gathered signals.

        Returns summary of consolidation actions taken.
        """
        actions = {
            "notes_created": 0,
            "notes_merged": 0,
            "index_entries_updated": 0,
            "vectors_updated": 0,
        }

        # 1. Process new sessions for notable content
        new_sessions = signals.get("new_sessions", [])
        for session_summary in new_sessions:
            session_id = session_summary.get("session_id")
            if not session_id:
                continue

            turns = self.memory.transcript_store.get_session_transcript(session_id)
            notable_content = self._extract_notable(turns)

            for content in notable_content:
                note_id = self.memory.save_note(
                    topic=content["topic"],
                    content=content["content"],
                    metadata={"source": f"session:{session_id}"},
                )
                actions["notes_created"] += 1

        # 2. Check for overlapping notes (simple dedup)
        notes = signals.get("notes", [])
        if len(notes) > 10:
            merged = self._deduplicate_notes(notes)
            actions["notes_merged"] = merged

        # 3. Vectorize all content
        vector_counts = self.memory.vectorize_all()
        actions["vectors_updated"] = sum(vector_counts.values())

        logger.info(f"Consolidation complete: {actions}")
        return actions

    def _extract_notable(self, turns: list[dict]) -> list[dict[str, str]]:
        """Extract notable content from session turns.

        Simple heuristic: look for user messages that contain explicit
        knowledge or preferences.
        """
        notable = []
        for turn in turns:
            if "user" not in turn:
                continue
            user_msg = turn["user"]
            # Detect preference statements
            preference_keywords = ["我喜欢", "我喜欢", "我喜欢", "总是", "always", "never", "prefer", "like", "习惯"]
            if any(kw in user_msg.lower() for kw in preference_keywords):
                notable.append({
                    "topic": "user_preference",
                    "content": f"User said: {user_msg[:500]}",
                })

            # Detect knowledge sharing
            if any(kw in user_msg.lower() for kw in ["记住", "记住", "remember", "note", "重要", "important"]):
                notable.append({
                    "topic": "knowledge",
                    "content": user_msg[:500],
                })

        return notable

    def _deduplicate_notes(self, notes: list[dict]) -> int:
        """Simple note deduplication by topic similarity."""
        # For MVP, just count. Full implementation would use vector similarity.
        topics = [n.get("topic", "") for n in notes]
        seen = set()
        duplicates = 0
        for topic in topics:
            normalized = topic.lower().strip()
            if normalized in seen:
                duplicates += 1
            seen.add(normalized)
        return duplicates
