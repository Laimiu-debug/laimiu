"""Unified memory manager - coordinates all three tiers + vector search."""

from __future__ import annotations

import logging
from typing import Any

from laimiu.config.settings import LaimiuConfig
from laimiu.constants import USER_PREFS_FILE
from laimiu.memory.detail import DetailStore
from laimiu.memory.index import MemoryIndex
from laimiu.memory.retriever import MemoryRetriever
from laimiu.memory.transcript import TranscriptStore
from laimiu.memory.vector_store import VectorStore

logger = logging.getLogger("laimiu.memory.manager")


class MemoryManager:
    """Unified interface for all memory operations.

    Coordinates:
    - Tier 1: MEMORY.md index (always loaded)
    - Tier 2: Topic notes (Markdown, vector-indexed)
    - Tier 3: Session transcripts (JSONL, vector-indexed)
    - ChromaDB vector search for semantic retrieval
    """

    def __init__(self, config: LaimiuConfig):
        self.config = config
        self.index = MemoryIndex()
        self.detail_store = DetailStore()
        self.transcript_store = TranscriptStore()
        self.vector_store = VectorStore()
        self.retriever = MemoryRetriever(
            self.vector_store,
        )
        self._session_count = 0

    def get_index(self) -> str:
        """Get the Tier 1 memory index (loaded every session)."""
        self.index.reload()
        return self.index.get()

    def search(self, query: str, max_results: int = 5) -> str:
        """Search all memories using hybrid retrieval."""
        return self.retriever.search(
            query,
            max_results=max_results,
            max_chars=self.config.memory.recall_max_chars,
        )

    def save_note(self, topic: str, content: str, metadata: dict | None = None) -> str:
        """Save a new topic note (Tier 2) and vectorize it."""
        note_id = self.detail_store.save_note(topic, content, metadata)
        # Store in vector DB
        self.vector_store.store_note(note_id, content, {"topic": topic, **(metadata or {})})
        # Update index
        self.index.add_entry("note", f"{topic}: {content[:80]}...", note_id)
        logger.info(f"Saved note: {note_id} ({topic})")
        return note_id

    def start_session(self) -> str:
        """Start a new session transcript."""
        self._session_count += 1
        return self.transcript_store.start_session()

    def save_turn(
        self,
        user_message: str,
        assistant_response: str,
        tools_used: list[dict] | None = None,
    ) -> None:
        """Save a conversation turn to transcript."""
        self.transcript_store.save_turn(user_message, assistant_response, tools_used)

    def save_tool_call(
        self,
        tool_name: str,
        args: dict,
        result: str,
        success: bool,
        reflection: dict | None = None,
    ) -> None:
        """Record a tool call in the transcript."""
        self.transcript_store.save_tool_call(tool_name, args, result, success, reflection)

    def end_session(self) -> None:
        """End the current session."""
        self.transcript_store.end_session()

    def get_user_preferences(self) -> str:
        """Load user preferences from user.md."""
        if USER_PREFS_FILE.exists():
            return USER_PREFS_FILE.read_text(encoding="utf-8")
        return ""

    def save_user_preference(self, preference: str) -> None:
        """Append a user preference."""
        existing = self.get_user_preferences()
        if not existing:
            existing = "# User Preferences\n\n"
        existing = existing.rstrip("\n") + "\n- " + preference + "\n"
        USER_PREFS_FILE.write_text(existing, encoding="utf-8")

    def vectorize_all(self) -> dict[str, int]:
        """Vectorize all existing notes and transcripts into ChromaDB.

        Returns counts of vectorized items.
        """
        counts = {"notes": 0, "transcripts": 0}

        # Vectorize notes
        for note_id, content in self.detail_store.get_all_content():
            self.vector_store.store_note(note_id, content, {})
            counts["notes"] += 1

        # Vectorize transcript chunks
        for chunk_id, content, metadata in self.transcript_store.get_all_chunks():
            self.vector_store.store_transcript_chunk(chunk_id, content, metadata)
            counts["transcripts"] += 1

        logger.info(f"Vectorized: {counts}")
        return counts

    def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        vector_stats = self.vector_store.get_stats()
        return {
            "index_lines": self.index.line_count,
            "notes": len(self.detail_store.list_notes()),
            "sessions": len(self.transcript_store.list_sessions()),
            "vector_notes": vector_stats.get("notes", 0),
            "vector_transcripts": vector_stats.get("transcripts", 0),
            "session_count": self._session_count,
        }
