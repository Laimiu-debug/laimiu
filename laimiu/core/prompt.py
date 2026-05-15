"""System prompt assembly - keeps prompts concise (≤2000 tokens)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from laimiu.constants import SOUL_FILE, SYSTEM_PROMPT_MAX_TOKENS
from laimiu.tools.base import ToolDefinition


def load_soul() -> str:
    """Load SOUL.md agent personality."""
    if SOUL_FILE.exists():
        return SOUL_FILE.read_text(encoding="utf-8")
    # Default personality
    return """# Laimiu

You are Laimiu, a personal AI assistant that learns and evolves with your user.

## Personality
- Direct, helpful, and proactive
- Remember user preferences and apply them
- When uncertain, ask rather than guess
- Use tools when they can help accomplish tasks

## Behavior
- Always respond in the same language the user uses
- Be concise but thorough
- Use memory_recall to check relevant past conversations when needed
- Track your own learning and improvement"""


def format_tool_list(tools: list[ToolDefinition]) -> str:
    """Format tool list as a concise description for the system prompt."""
    lines = ["## Available Tools\n"]
    for tool in tools:
        lines.append(f"- **{tool.name}**: {tool.description}")
    return "\n".join(lines)


def build_system_prompt(
    soul: str | None = None,
    memory_index: str = "",
    tool_list: list[ToolDefinition] | None = None,
    user_prefs: str = "",
) -> str:
    """Build the system prompt from components.

    Strictly controlled to fit within SYSTEM_PROMPT_MAX_TOKENS (~2000 tokens).
    Components:
    - SOUL.md (personality): ~500 tokens
    - MEMORY.md index (≤200 lines): ~800 tokens
    - Tool list: ~500 tokens
    """
    parts = []

    # 1. Soul (personality)
    if soul is None:
        soul = load_soul()
    parts.append(soul)

    # 2. User preferences (brief)
    if user_prefs:
        # Truncate to keep concise
        prefs_text = user_prefs[:500]
        parts.append(f"## User Preferences\n{prefs_text}")

    # 3. Memory index (Tier 1)
    if memory_index and memory_index.strip():
        # Truncate if too long
        lines = memory_index.strip().split("\n")
        if len(lines) > 50:  # Keep index concise in prompt
            lines = lines[:50]
        parts.append("## Memory Index\n" + "\n".join(lines))

    # 4. Tool list
    if tool_list:
        parts.append(format_tool_list(tool_list))

    return "\n\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (1 token ≈ 4 chars for English, ~2 chars for Chinese)."""
    # Simple heuristic
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_chars = len(text) - chinese_chars
    return chinese_chars // 2 + other_chars // 4
