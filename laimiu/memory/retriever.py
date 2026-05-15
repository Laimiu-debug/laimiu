"""Memory retriever - hybrid search combining vector and keyword matching."""

from __future__ import annotations

import logging
from typing import Any

from laimiu.memory.vector_store import VectorStore

logger = logging.getLogger("laimiu.memory.retriever")


class MemoryRetriever:
    """Hybrid memory retrieval: vector semantic search + keyword matching.

    Ranks results using a combined score:
      final_score = alpha * vector_score + (1 - alpha) * keyword_score
    """

    def __init__(self, vector_store: VectorStore, alpha: float = 0.7):
        """
        Args:
            vector_store: The ChromaDB vector store.
            alpha: Weight for vector vs keyword scoring (0.7 = 70% vector, 30% keyword).
        """
        self.vector_store = vector_store
        self.alpha = alpha

    def search(
        self,
        query: str,
        max_results: int = 5,
        max_chars: int = 2000,
    ) -> str:
        """Search memories and return formatted results.

        Returns a human-readable string with the most relevant memories.
        """
        # Vector search
        vector_results = self.vector_store.search(
            query, max_results=max_results * 2, max_chars=max_chars * 2
        )

        if not vector_results:
            return "No relevant memories found."

        # Keyword scoring
        query_terms = set(query.lower().split())
        scored_results = []
        for result in vector_results:
            content_lower = result["content"].lower()
            # Count matching terms
            keyword_hits = sum(1 for term in query_terms if term in content_lower)
            keyword_score = keyword_hits / max(len(query_terms), 1)

            # Vector distance -> score (lower distance = higher score)
            vector_score = 1.0 - result.get("distance", 0.5)

            # Combined score
            final_score = self.alpha * vector_score + (1 - self.alpha) * keyword_score
            scored_results.append((final_score, result))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Format output within char budget
        output_parts = []
        total_chars = 0
        for score, result in scored_results[:max_results]:
            source = result.get("source", "unknown")
            content = result["content"]
            # Truncate individual results if needed
            if total_chars + len(content) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break
            output_parts.append(f"[{source}] (score: {score:.2f})\n{content}")
            total_chars += len(content)

        if not output_parts:
            return "No relevant memories found."

        return "\n\n---\n\n".join(output_parts)

    def search_raw(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Return raw search results with scoring."""
        vector_results = self.vector_store.search(
            query, max_results=max_results * 2, max_chars=10000
        )

        query_terms = set(query.lower().split())
        scored = []
        for result in vector_results:
            content_lower = result["content"].lower()
            keyword_hits = sum(1 for term in query_terms if term in content_lower)
            keyword_score = keyword_hits / max(len(query_terms), 1)
            vector_score = 1.0 - result.get("distance", 0.5)
            final_score = self.alpha * vector_score + (1 - self.alpha) * keyword_score
            scored.append({**result, "score": final_score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:max_results]
