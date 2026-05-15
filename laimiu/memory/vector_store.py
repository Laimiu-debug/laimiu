"""ChromaDB vector store for semantic memory search."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from laimiu.constants import CHROMA_DIR

logger = logging.getLogger("laimiu.memory.vector_store")

# Serialize all ChromaDB writes — PersistentClient is not thread-safe.
_chroma_lock = threading.Lock()


class VectorStore:
    """ChromaDB-based vector store for semantic memory search.

    Stores embeddings for Tier 2 notes and Tier 3 transcript chunks.
    Supports hybrid search: vector similarity + keyword matching.
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or CHROMA_DIR
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._notes_collection = None
        self._transcripts_collection = None

    def _get_client(self):
        """Lazy-init ChromaDB client."""
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.db_path))
        return self._client

    def _get_notes_collection(self):
        """Get or create the notes collection."""
        if self._notes_collection is None:
            client = self._get_client()
            self._notes_collection = client.get_or_create_collection(
                name="notes",
                metadata={"hnsw:space": "cosine"},
            )
        return self._notes_collection

    def _get_transcripts_collection(self):
        """Get or create the transcripts collection."""
        if self._transcripts_collection is None:
            client = self._get_client()
            self._transcripts_collection = client.get_or_create_collection(
                name="transcripts",
                metadata={"hnsw:space": "cosine"},
            )
        return self._transcripts_collection

    def store_note(self, note_id: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Store a Tier 2 note's vector embedding."""
        collection = self._get_notes_collection()
        with _chroma_lock:
            # ChromaDB auto-generates embeddings if no embedding function is provided
            collection.upsert(
                ids=[note_id],
                documents=[content],
                metadatas=[metadata or {}],
            )
        logger.debug(f"Stored note vector: {note_id}")

    def store_transcript_chunk(self, chunk_id: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Store a Tier 3 transcript chunk's vector embedding."""
        collection = self._get_transcripts_collection()
        with _chroma_lock:
            collection.upsert(
                ids=[chunk_id],
                documents=[content],
                metadatas=[metadata or {}],
            )
        logger.debug(f"Stored transcript chunk: {chunk_id}")

    def search(
        self,
        query: str,
        max_results: int = 5,
        max_chars: int = 2000,
        collection: str = "all",
    ) -> list[dict[str, Any]]:
        """Search memories using vector similarity.

        Args:
            query: Search query text.
            max_results: Maximum number of results.
            max_chars: Hard limit on total returned characters.
            collection: "notes", "transcripts", or "all".

        Returns:
            List of search results with content and metadata.
        """
        results = []
        total_chars = 0

        if collection in ("notes", "all"):
            notes_coll = self._get_notes_collection()
            if notes_coll.count() > 0:
                try:
                    note_results = notes_coll.query(
                        query_texts=[query],
                        n_results=min(max_results, notes_coll.count()),
                    )
                    if note_results["documents"]:
                        for i, doc in enumerate(note_results["documents"][0]):
                            dist = note_results["distances"][0][i] if note_results["distances"] else 0
                            meta = note_results["metadatas"][0][i] if note_results["metadatas"] else {}
                            if total_chars + len(doc) <= max_chars:
                                results.append({
                                    "content": doc,
                                    "metadata": meta,
                                    "distance": dist,
                                    "source": "notes",
                                })
                                total_chars += len(doc)
                except Exception as e:
                    logger.error(f"Notes search failed: {e}")

        if collection in ("transcripts", "all") and total_chars < max_chars:
            trans_coll = self._get_transcripts_collection()
            if trans_coll.count() > 0:
                remaining = max_results - len(results)
                if remaining > 0:
                    try:
                        trans_results = trans_coll.query(
                            query_texts=[query],
                            n_results=min(remaining, trans_coll.count()),
                        )
                        if trans_results["documents"]:
                            for i, doc in enumerate(trans_results["documents"][0]):
                                dist = trans_results["distances"][0][i] if trans_results["distances"] else 0
                                meta = trans_results["metadatas"][0][i] if trans_results["metadatas"] else {}
                                if total_chars + len(doc) <= max_chars:
                                    results.append({
                                        "content": doc,
                                        "metadata": meta,
                                        "distance": dist,
                                        "source": "transcripts",
                                    })
                                    total_chars += len(doc)
                    except Exception as e:
                        logger.error(f"Transcript search failed: {e}")

        return results

    def search_notes(self, query: str, max_results: int = 5) -> list[dict]:
        """Convenience: search only notes."""
        return self.search(query, max_results=max_results, collection="notes")

    def delete_note(self, note_id: str) -> None:
        """Remove a note from the vector store."""
        collection = self._get_notes_collection()
        try:
            collection.delete(ids=[note_id])
        except Exception:
            pass  # May not exist

    def get_stats(self) -> dict[str, int]:
        """Get collection sizes."""
        stats = {}
        try:
            notes_coll = self._get_notes_collection()
            stats["notes"] = notes_coll.count()
        except Exception:
            stats["notes"] = 0
        try:
            trans_coll = self._get_transcripts_collection()
            stats["transcripts"] = trans_coll.count()
        except Exception:
            stats["transcripts"] = 0
        return stats
