"""Pattern tracker - progressive memory strength model for procedural extraction.

Replaces the fixed occurrence-threshold with a human-like progressive memory model:
- imprint → short_term → long_term → procedural
- Strength decays over time (Ebbinghaus-inspired) but is reinforced by repetition
- Success/failure adjusts the reinforcement rate
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from laimiu.constants import PATTERNS_FILE
from laimiu.core.reflection import ReflectionResult
from laimiu.tools.base import ToolResult
from laimiu.utils.io import atomic_write

logger = logging.getLogger("laimiu.procedural.tracker")

# --- Progressive memory constants ---
BASE_INCREMENT = 0.15       # First encounter adds 0.15
HALF_LIFE_DAYS = 7.0        # Memory half-life in days
SUCCESS_BONUS = 0.2         # Extra boost on successful execution
FAILURE_PENALTY = 0.3       # Penalty on failed execution

# Memory level thresholds
LEVEL_THRESHOLDS = {
    "imprint": 0.0,
    "short_term": 0.2,
    "long_term": 0.5,
    "procedural": 0.8,
}


def _compute_level(strength: float) -> str:
    """Determine memory level from strength value."""
    if strength >= LEVEL_THRESHOLDS["procedural"]:
        return "procedural"
    if strength >= LEVEL_THRESHOLDS["long_term"]:
        return "long_term"
    if strength >= LEVEL_THRESHOLDS["short_term"]:
        return "short_term"
    return "imprint"


def _compute_strength(
    current_strength: float,
    occurrence: int,
    dt_days: float,
    was_success: bool,
) -> float:
    """Compute new memory strength after an encounter.

    Args:
        current_strength: Current strength (0.0-1.0)
        occurrence: Total number of times this pattern has been seen (including this time)
        dt_days: Days since last encounter
        was_success: Whether the tool call succeeded

    Returns:
        Updated strength (0.0-1.0)
    """
    # 1. Time-based decay (Ebbinghaus)
    decay = math.exp(-dt_days / HALF_LIFE_DAYS) if dt_days > 0 else 1.0

    # 2. Reinforcement increment grows with practice
    delta = BASE_INCREMENT * (1.0 + 0.5 * max(0, occurrence - 1))
    # Cap delta so it can't exceed 1.0 in one shot
    delta = min(delta, 0.8)

    # 3. New strength = decayed old + new increment
    new_strength = current_strength * decay + delta

    # 4. Success/failure adjustment
    if was_success:
        new_strength += SUCCESS_BONUS
    else:
        new_strength -= FAILURE_PENALTY

    return max(0.0, min(1.0, new_strength))


@dataclass
class Pattern:
    """A detected repeated operation pattern with progressive memory strength."""

    tool_name: str
    args_signature: str  # Hash of normalized args
    # Progressive memory model
    strength: float = 0.0          # 0.0 ~ 1.0, memory strength
    level: str = "imprint"         # imprint → short_term → long_term → procedural
    occurrence: int = 0
    first_seen: str = ""
    last_seen: str = ""
    success_rate: float = 0.0
    examples: list[dict[str, Any]] = field(default_factory=list)
    reinforcement_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "args_signature": self.args_signature,
            "strength": round(self.strength, 4),
            "level": self.level,
            "occurrence": self.occurrence,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "success_rate": round(self.success_rate, 4),
            "examples": self.examples[:3],
            "reinforcement_history": self.reinforcement_history[-10:],
        }


class PatternTracker:
    """Tracks tool call patterns with progressive memory strength.

    Instead of a fixed occurrence threshold, patterns build up memory strength
    through repeated encounters. Time decay causes unused patterns to fade,
    while frequent use reinforces them — mimicking human procedural memory.
    """

    def __init__(
        self,
        extract_strength: float = 0.6,
        patterns_file: Path | None = None,
    ):
        self.extract_strength = extract_strength
        self.patterns_file = patterns_file or PATTERNS_FILE
        self._patterns: dict[str, Pattern] = {}
        self._load_patterns()

    def _normalize_args(self, args: dict[str, Any]) -> str:
        """Create a normalized hash of tool arguments to group similar calls."""
        normalized = {}
        for key, value in args.items():
            if isinstance(value, str):
                normalized[key] = f"str({len(value)})"
            elif isinstance(value, (int, float)):
                normalized[key] = type(value).__name__
            else:
                normalized[key] = type(value).__name__
        return hashlib.md5(json.dumps(normalized, sort_keys=True).encode()).hexdigest()[:12]

    def record(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
        reflection: ReflectionResult,
    ) -> None:
        """Record a tool call for pattern tracking with progressive strength."""
        args_sig = self._normalize_args(args)
        key = f"{tool_name}:{args_sig}"
        now = datetime.now()
        now_iso = now.isoformat()

        if key not in self._patterns:
            self._patterns[key] = Pattern(
                tool_name=tool_name,
                args_signature=args_sig,
                first_seen=now_iso,
            )

        pattern = self._patterns[key]
        pattern.occurrence += 1

        # Compute time delta in days
        if pattern.last_seen:
            try:
                last = datetime.fromisoformat(pattern.last_seen)
                dt_days = (now - last).total_seconds() / 86400.0
            except (ValueError, TypeError):
                dt_days = 0.0
        else:
            dt_days = 0.0

        # Compute new strength
        old_strength = pattern.strength
        pattern.strength = _compute_strength(
            old_strength, pattern.occurrence, dt_days, result.success,
        )
        pattern.level = _compute_level(pattern.strength)
        pattern.last_seen = now_iso

        # Update success rate
        if result.success:
            successes = pattern.success_rate * (pattern.occurrence - 1) + 1
        else:
            successes = pattern.success_rate * (pattern.occurrence - 1)
        pattern.success_rate = successes / pattern.occurrence

        # Store example (keep max 5)
        if len(pattern.examples) < 5:
            pattern.examples.append({
                "args": {k: str(v)[:100] for k, v in args.items()},
                "success": result.success,
                "timestamp": now_iso,
            })

        # Record reinforcement event (keep last 10)
        pattern.reinforcement_history.append({
            "at": now_iso,
            "old_strength": round(old_strength, 4),
            "new_strength": round(pattern.strength, 4),
            "success": result.success,
            "dt_days": round(dt_days, 2),
        })
        if len(pattern.reinforcement_history) > 10:
            pattern.reinforcement_history = pattern.reinforcement_history[-10:]

        self._save_patterns()

        logger.debug(
            f"Pattern {key}: strength {old_strength:.2f} → {pattern.strength:.2f} "
            f"({pattern.level}, occ={pattern.occurrence})"
        )

    def get_extractable_patterns(self) -> list[Pattern]:
        """Get patterns whose strength exceeds the extraction threshold."""
        extractable = []
        for pattern in self._patterns.values():
            if (
                pattern.strength >= self.extract_strength
                and pattern.success_rate >= 0.5
                and pattern.occurrence >= 2  # Minimum encounters
            ):
                extractable.append(pattern)
        return extractable

    def get_all_patterns(self) -> list[Pattern]:
        """Get all tracked patterns."""
        return list(self._patterns.values())

    def clear_pattern(self, key: str) -> None:
        """Remove a pattern after it's been extracted."""
        self._patterns.pop(key, None)
        self._save_patterns()

    def _load_patterns(self) -> None:
        """Load patterns from disk."""
        if not self.patterns_file.exists():
            return
        try:
            with open(self.patterns_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    key = f"{data['tool_name']}:{data['args_signature']}"
                    self._patterns[key] = Pattern(
                        tool_name=data["tool_name"],
                        args_signature=data["args_signature"],
                        strength=data.get("strength", 0.0),
                        level=data.get("level", "imprint"),
                        occurrence=data.get("occurrence", 0),
                        first_seen=data.get("first_seen", ""),
                        last_seen=data.get("last_seen", ""),
                        success_rate=data.get("success_rate", 0.0),
                        examples=data.get("examples", []),
                        reinforcement_history=data.get("reinforcement_history", []),
                    )
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")

    def _save_patterns(self) -> None:
        """Save patterns to disk atomically."""
        self.patterns_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            lines = []
            for pattern in self._patterns.values():
                lines.append(json.dumps(pattern.to_dict(), ensure_ascii=False))
            content = "\n".join(lines) + "\n" if lines else ""
            atomic_write(self.patterns_file, content)
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")
