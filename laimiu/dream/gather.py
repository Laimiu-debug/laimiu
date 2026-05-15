"""Dream Phase 2: Gather - collect signals from all memory tiers."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("laimiu.dream.gather")


class Gather:
    """Phase 2: Gather - collect raw data from all memory tiers."""

    def gather(self, memory_manager: Any, orientation: dict[str, Any]) -> dict[str, Any]:
        """Collect signals from memory.

        Returns:
            Dictionary with gathered data from each tier.
        """
        signals: dict[str, Any] = {}

        # Gather from Tier 1 (index)
        signals["memory_index"] = memory_manager.get_index()

        # Gather from Tier 2 (notes)
        notes = memory_manager.detail_store.list_notes()
        signals["notes"] = notes
        signals["note_count"] = len(notes)

        # Gather from Tier 3 (transcripts)
        sessions = memory_manager.transcript_store.list_sessions()
        signals["sessions"] = sessions[-10:]  # Last 10 sessions
        signals["session_count"] = len(sessions)

        # Identify new sessions since last dream
        state = orientation.get("state", {})
        sessions_since = state.get("sessions_since_dream", 0)
        if sessions_since > 0 and len(sessions) >= sessions_since:
            signals["new_sessions"] = sessions[:sessions_since]
        else:
            signals["new_sessions"] = []

        logger.info(
            f"Gathered: {signals['note_count']} notes, "
            f"{signals['session_count']} sessions, "
            f"{len(signals.get('new_sessions', []))} new"
        )

        return signals
