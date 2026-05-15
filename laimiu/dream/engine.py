"""Dream Engine - 5-phase autonomous memory processing."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from laimiu.config.settings import LaimiuConfig
from laimiu.constants import DREAM_LOG_FILE
from laimiu.dream.consolidate import Consolidate
from laimiu.dream.extract import ProceduralExtractor
from laimiu.dream.gather import Gather
from laimiu.dream.orient import Orient
from laimiu.dream.prune import Prune
from laimiu.memory.manager import MemoryManager
from laimiu.procedural.engine import ProceduralEngine
from laimiu.safety.guardian import Guardian

logger = logging.getLogger("laimiu.dream.engine")


class DreamEngine:
    """5-phase Dream engine for autonomous memory processing.

    Phases:
    1. Orient - assess current state
    2. Gather - collect signals from all tiers
    3. Consolidate - integrate and organize memories
    4. Prune - keep the index concise
    5. Extract - create procedural tools from patterns
    """

    def __init__(
        self,
        config: LaimiuConfig,
        memory: MemoryManager,
        procedural_engine: ProceduralEngine | None = None,
        router: Any = None,
    ):
        self.config = config
        self.memory = memory
        self.procedural_engine = procedural_engine
        self.router = router

        self.orient = Orient()
        self.gather = Gather()
        self.consolidate = Consolidate(memory, router)
        self.prune = Prune(memory)
        self.extractor = (
            ProceduralExtractor(procedural_engine) if procedural_engine else None
        )

    async def dream(self) -> dict[str, Any]:
        """Run a full dream cycle.

        Returns summary of all phases.
        """
        if not self.config.dream.enabled:
            logger.info("Dream engine disabled")
            return {"skipped": True, "reason": "disabled"}

        logger.info("Starting dream cycle...")

        # Pre-mutation snapshot
        guardian = Guardian()
        snapshot_tag = guardian.pre_mutation_snapshot()

        start_time = datetime.now()
        results: dict[str, Any] = {
            "started_at": start_time.isoformat(),
            "phases": {},
        }

        try:
            # Phase 1: Orient
            orientation = self.orient.orient(self.memory, self.procedural_engine)
            results["phases"]["orient"] = orientation
            logger.info(f"Phase 1 (Orient): {orientation['memory_stats']}")

            # Phase 2: Gather
            signals = self.gather.gather(self.memory, orientation)
            results["phases"]["gather"] = {
                "note_count": signals.get("note_count", 0),
                "session_count": signals.get("session_count", 0),
                "new_sessions": len(signals.get("new_sessions", [])),
            }
            logger.info(f"Phase 2 (Gather): {signals['note_count']} notes, {signals['session_count']} sessions")

            # Phase 3: Consolidate
            consolidation = await self.consolidate.consolidate(signals)
            results["phases"]["consolidate"] = consolidation
            logger.info(f"Phase 3 (Consolidate): {consolidation}")

            # Phase 4: Prune
            pruning = self.prune.prune()
            results["phases"]["prune"] = pruning
            logger.info(f"Phase 4 (Prune): {pruning}")

            # Phase 5: Extract (procedural memory)
            if self.extractor and self.config.procedural.enabled:
                extraction = await self.extractor.extract()
                results["phases"]["extract"] = extraction
                logger.info(f"Phase 5 (Extract): {len(extraction.get('tools_generated', []))} tools created")

            # Update state
            self.orient.update_state({
                "last_dream": start_time.isoformat(),
                "sessions_since_dream": 0,
                "total_sessions": orientation["memory_stats"].get("sessions", 0),
            })

        except Exception as e:
            logger.error(f"Dream cycle failed: {e}")
            results["error"] = str(e)

        # Log the dream
        end_time = datetime.now()
        results["completed_at"] = end_time.isoformat()
        results["duration_seconds"] = (end_time - start_time).total_seconds()
        self._log_dream(results)

        logger.info(f"Dream cycle completed in {results['duration_seconds']:.1f}s")
        return results

    def should_dream(self) -> bool:
        """Check if it's time for a dream cycle."""
        return self.orient.should_dream(self.config)

    def increment_sessions(self) -> None:
        """Increment the session counter (called when a session ends)."""
        state = self.orient.get_state()
        state["sessions_since_dream"] = state.get("sessions_since_dream", 0) + 1
        self.orient.update_state(state)

    def _log_dream(self, results: dict[str, Any]) -> None:
        """Append dream results to the dream log."""
        DREAM_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(DREAM_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(results, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log dream: {e}")
