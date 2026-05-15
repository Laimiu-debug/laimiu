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
    provider_name: str = "",
    model_name: str = "",
) -> str:
    """Build the system prompt from components.

    Strictly controlled to fit within SYSTEM_PROMPT_MAX_TOKENS (~2000 tokens).
    """
    parts = []

    # 1. Soul (personality)
    if soul is None:
        soul = load_soul()
    parts.append(soul)

    # 2. Identity — tell the model WHO it is, so it doesn't hallucinate
    identity_lines = ["## Identity"]
    identity_lines.append(
        "Your name is Laimiu. You are an independent AI assistant with your own memory and learning system."
    )
    if provider_name or model_name:
        identity_lines.append(
            f"Currently powered by: {provider_name}/{model_name}. "
            "You are NOT Claude, ChatGPT, or any other assistant — you are Laimiu."
        )
    else:
        identity_lines.append(
            "You are NOT Claude, ChatGPT, or any other assistant — you are Laimiu."
        )
    parts.append("\n".join(identity_lines))

    # 3. Conversation rules
    rules = """## Conversation Rules
- Do NOT repeat greetings or self-introductions — just answer the user directly
- Do NOT list your capabilities unless the user explicitly asks
- Continue the conversation naturally, as if you remember everything from previous turns
- Respond in the same language the user uses (Chinese → Chinese, English → English)
- Be concise — avoid unnecessary filler"""
    parts.append(rules)

    # 4. User preferences (brief)
    if user_prefs:
        prefs_text = user_prefs[:500]
        parts.append(f"## User Preferences\n{prefs_text}")

    # 5. Memory index (Tier 1)
    if memory_index and memory_index.strip():
        lines = memory_index.strip().split("\n")
        if len(lines) > 50:
            lines = lines[:50]
        parts.append("## Memory Index\n" + "\n".join(lines))

    # 6. Tool list
    if tool_list:
        parts.append(format_tool_list(tool_list))

    return "\n\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (1 token ≈ 4 chars for English, ~2 chars for Chinese)."""
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_chars = len(text) - chinese_chars
    return chinese_chars // 2 + other_chars // 4
