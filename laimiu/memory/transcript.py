"""Tier 3: Session transcripts (JSONL, searchable, never loaded to context)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from laimiu.constants import TRANSCRIPTS_DIR

logger = logging.getLogger("laimiu.memory.transcript")


class TranscriptStore:
    """Manages Tier 3 session transcripts.

    Transcripts are append-only JSONL files containing full conversation history.
    They are vectorized for search but never loaded into the LLM context.
    """

    def __init__(self, transcripts_dir: Path | None = None):
        self.transcripts_dir = transcripts_dir or TRANSCRIPTS_DIR
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self._current_file: Path | None = None
        self._session_id: str | None = None

    def start_session(self, session_id: str | None = None) -> str:
        """Start a new session transcript."""
        self._session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_file = self.transcripts_dir / f"session_{self._session_id}.jsonl"
        return self._session_id

    def save_turn(
        self,
        user_message: str,
        assistant_response: str,
        tools_used: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a conversation turn."""
        if self._current_file is None:
            self.start_session()

        turn = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self._session_id,
            "user": user_message,
            "assistant": assistant_response,
        }
        if tools_used:
            turn["tools_used"] = tools_used
        if metadata:
            turn["metadata"] = metadata

        with open(self._current_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")

    def save_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str,
        success: bool,
        reflection: dict[str, Any] | None = None,
    ) -> None:
        """Record a tool call within the current session."""
        if self._current_file is None:
            self.start_session()

        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self._session_id,
            "type": "tool_call",
            "tool": tool_name,
            "args": args,
            "result_summary": result[:500] if result else "",
            "success": success,
        }
        if reflection:
            entry["reflection"] = reflection

        with open(self._current_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def end_session(self) -> None:
        """End the current session."""
        if self._current_file is None:
            return
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self._session_id,
            "type": "session_end",
        }
        with open(self._current_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._current_file = None
        self._session_id = None

    def get_session_transcript(self, session_id: str) -> list[dict]:
        """Read a full session transcript."""
        path = self.transcripts_dir / f"session_{session_id}.jsonl"
        if not path.exists():
            return []
        turns = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    turns.append(json.loads(line))
        return turns

    def list_sessions(self) -> list[dict]:
        """List all sessions with metadata."""
        sessions = []
        for jsonl_file in sorted(self.transcripts_dir.glob("session_*.jsonl"), reverse=True):
            session_id = jsonl_file.stem.replace("session_", "")
            turns = []
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        turns.append(json.loads(line))

            user_turns = [t for t in turns if "user" in t]
            sessions.append({
                "session_id": session_id,
                "turns": len(user_turns),
                "first_turn": turns[0]["timestamp"] if turns else None,
                "last_turn": turns[-1]["timestamp"] if turns else None,
            })
        return sessions

    def get_all_chunks(self, chunk_size: int = 3) -> list[tuple[str, str, dict]]:
        """Get transcript chunks for vectorization.

        Returns list of (chunk_id, content_text, metadata) tuples.
        """
        chunks = []
        for jsonl_file in sorted(self.transcripts_dir.glob("session_*.jsonl")):
            session_id = jsonl_file.stem.replace("session_", "")
            turns = []
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        turns.append(json.loads(line))

            # Chunk consecutive turns together
            for i in range(0, len(turns), chunk_size):
                chunk = turns[i:i + chunk_size]
                text_parts = []
                for t in chunk:
                    if "user" in t:
                        text_parts.append(f"User: {t['user']}")
                    if "assistant" in t:
                        text_parts.append(f"Assistant: {t['assistant'][:500]}")
                    if t.get("type") == "tool_call":
                        text_parts.append(f"Tool({t['tool']}): {t.get('result_summary', '')[:200]}")
                if text_parts:
                    chunk_text = "\n".join(text_parts)
                    chunk_id = f"{session_id}_chunk_{i // chunk_size}"
                    chunks.append((chunk_id, chunk_text, {"session": session_id, "index": i}))
        return chunks
