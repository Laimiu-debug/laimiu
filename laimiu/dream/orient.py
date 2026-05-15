"""Dream Phase 1: Orient - assess current state."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from laimiu.constants import DREAM_STATE_FILE
from laimiu.utils.io import atomic_write

logger = logging.getLogger("laimiu.dream.orient")


class Orient:
    """Phase 1: Orient - assess the current state of memory and sessions."""

    def __init__(self, state_file: Path | None = None):
        self.state_file = state_file or DREAM_STATE_FILE

    def get_state(self) -> dict[str, Any]:
        """Load the current dream state."""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "last_dream": None,
            "sessions_since_dream": 0,
            "total_sessions": 0,
            "last_orientation": None,
        }

    def update_state(self, updates: dict[str, Any]) -> None:
        """Update dream state."""
        state = self.get_state()
        state.update(updates)
        state["last_orientation"] = datetime.now().isoformat()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            self.state_file,
            json.dumps(state, indent=2, ensure_ascii=False),
        )

    def should_dream(self, config: Any) -> bool:
        """Check if it's time for a dream cycle."""
        state = self.get_state()

        # Never dreamed before
        if state.get("last_dream") is None:
            return state.get("sessions_since_dream", 0) >= config.dream.trigger_after_sessions

        # Check session count trigger
        if state.get("sessions_since_dream", 0) >= config.dream.trigger_after_sessions:
            return True

        # Check time trigger
        if config.dream.trigger_after_hours > 0 and state.get("last_dream"):
            last = datetime.fromisoformat(state["last_dream"])
            hours_since = (datetime.now() - last).total_seconds() / 3600
            if hours_since >= config.dream.trigger_after_hours:
                return True

        return False

    def orient(self, memory_manager: Any, procedural_engine: Any | None = None) -> dict[str, Any]:
        """Gather orientation data about the current state.

        Returns a summary of what needs processing.
        """
        state = self.get_state()
        memory_stats = memory_manager.get_stats()

        orientation = {
            "state": state,
            "memory_stats": memory_stats,
            "needs_consolidation": memory_stats.get("notes", 0) > 5,
            "needs_procedural_extract": False,
            "timestamp": datetime.now().isoformat(),
        }

        if procedural_engine:
            proc_stats = procedural_engine.get_stats()
            orientation["procedural_stats"] = proc_stats
            orientation["needs_procedural_extract"] = proc_stats.get("extractable_patterns", 0) > 0

        return orientation
