"""Dataset builder - constructs fine-tuning datasets from conversation transcripts."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from laimiu.constants import TRANSCRIPTS_DIR

logger = logging.getLogger("laimiu.adaptation.dataset_builder")


class DatasetBuilder:
    """Builds fine-tuning datasets from conversation history.

    Filters for high-quality conversations (user satisfied, task completed,
    no corrections). Converts to Alpaca training format.
    """

    def build(
        self,
        transcripts_dir: Path | None = None,
        output_path: Path | None = None,
        min_quality_score: float = 0.7,
    ) -> dict[str, Any]:
        """Build a fine-tuning dataset.

        Args:
            transcripts_dir: Directory containing session JSONL files.
            output_path: Where to write the output dataset.
            min_quality_score: Minimum quality threshold for examples.

        Returns:
            Summary of the build process.
        """
        transcripts_dir = transcripts_dir or TRANSCRIPTS_DIR
        if output_path is None:
            output_path = transcripts_dir.parent / "adaptation_dataset.jsonl"

        if not transcripts_dir.exists():
            return {"status": "no_data", "examples": 0}

        # Collect all conversation turns
        all_turns = self._load_all_turns(transcripts_dir)

        # Filter for quality
        quality_turns = [t for t in all_turns if self._assess_quality(t) >= min_quality_score]

        # Convert to Alpaca format
        dataset = []
        for turn in quality_turns:
            example = self._to_alpaca_format(turn)
            if example:
                dataset.append(example)

        # Deduplicate
        dataset = self._deduplicate(dataset)

        # Redact sensitive information
        dataset = self._redact_secrets(dataset)

        # Write output
        if output_path and dataset:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for example in dataset:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")

        return {
            "status": "success",
            "total_turns": len(all_turns),
            "quality_turns": len(quality_turns),
            "examples": len(dataset),
            "output_path": str(output_path) if output_path else None,
        }

    def _load_all_turns(self, transcripts_dir: Path) -> list[dict]:
        """Load all conversation turns from JSONL files."""
        turns = []
        for jsonl_file in sorted(transcripts_dir.glob("session_*.jsonl")):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if "user" in entry and "assistant" in entry:
                            turns.append(entry)
                    except json.JSONDecodeError:
                        continue
        return turns

    def _assess_quality(self, turn: dict) -> float:
        """Assess the quality of a conversation turn.

        Heuristics:
        - Longer, more substantive exchanges score higher
        - Turns with tool usage indicate real tasks
        - Short exchanges score lower
        """
        score = 0.5

        # Substantive content
        if len(turn.get("assistant", "")) > 100:
            score += 0.2

        # Tool usage indicates real task
        if turn.get("tools_used"):
            score += 0.1

        # Not too short
        if len(turn.get("user", "")) > 20:
            score += 0.1

        return min(score, 1.0)

    def _to_alpaca_format(self, turn: dict) -> dict[str, str] | None:
        """Convert a turn to Alpaca fine-tuning format."""
        user_msg = turn.get("user", "").strip()
        assistant_msg = turn.get("assistant", "").strip()

        if not user_msg or not assistant_msg:
            return None

        return {
            "instruction": "你是 Laimiu，用户的个人 AI 助手。直接、有帮助、主动。",
            "input": user_msg,
            "output": assistant_msg,
        }

    def _deduplicate(self, dataset: list[dict]) -> list[dict]:
        """Remove duplicate examples based on input."""
        seen = set()
        unique = []
        for example in dataset:
            key = example.get("input", "")[:200]
            if key not in seen:
                seen.add(key)
                unique.append(example)
        return unique

    def _redact_secrets(self, dataset: list[dict]) -> list[dict]:
        """Remove sensitive information from examples."""
        secret_patterns = [
            (r'(api[_-]?key["\s:=]+)["\']?\w{10,}["\']?', r'\1[REDACTED]'),
            (r'(password["\s:=]+)["\']?\S+["\']?', r'\1[REDACTED]'),
            (r'(token["\s:=]+)["\']?\S+["\']?', r'\1[REDACTED]'),
            (r'(secret["\s:=]+)["\']?\S+["\']?', r'\1[REDACTED]'),
        ]
        redacted = []
        for example in dataset:
            ex = example.copy()
            for key in ("input", "output"):
                text = ex.get(key, "")
                for pattern, replacement in secret_patterns:
                    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                ex[key] = text
            redacted.append(ex)
        return redacted
