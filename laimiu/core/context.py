"""Context management - token budget and conversation compression."""

from __future__ import annotations

import logging
from typing import Any

from laimiu.providers.base import Message

logger = logging.getLogger("laimiu.core.context")

# Approximate token limits
MAX_CONTEXT_TOKENS = 100_000
RESERVED_FOR_SYSTEM = 3_000
RESERVED_FOR_RESPONSE = 4_000
MAX_MESSAGES_IN_CONTEXT = 100


def estimate_message_tokens(msg: Message) -> int:
    """Estimate token count for a message."""
    total = len(msg.content) // 3  # Rough estimate
    if msg.tool_calls:
        for tc in msg.tool_calls:
            total += len(str(tc)) // 3
    return max(total, 1)


def estimate_messages_tokens(messages: list[Message]) -> int:
    """Estimate total tokens for a message list."""
    return sum(estimate_message_tokens(m) for m in messages)


class ContextManager:
    """Manages conversation context within token budget.

    When context gets too large:
    1. First, drop oldest tool call results (they're verbose)
    2. Then, summarize early conversation turns
    """

    def __init__(self, max_tokens: int = MAX_CONTEXT_TOKENS):
        self.max_tokens = max_tokens
        self.budget = max_tokens - RESERVED_FOR_SYSTEM - RESERVED_FOR_RESPONSE

    def should_compress(self, messages: list[Message]) -> bool:
        """Check if context needs compression."""
        tokens = estimate_messages_tokens(messages)
        return tokens > self.budget or len(messages) > MAX_MESSAGES_IN_CONTEXT

    def compress(self, messages: list[Message]) -> list[Message]:
        """Compress messages to fit within budget.

        Strategy:
        1. Keep system message (first)
        2. Keep last N messages
        3. Summarize the rest into a single message
        """
        if not messages:
            return messages

        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        if len(non_system) <= 6:
            return messages  # Too few to compress

        # Keep last 6 messages intact
        recent = non_system[-6:]
        older = non_system[:-6]

        # Create summary of older messages
        summary_parts = []
        for msg in older:
            if msg.role == "user":
                summary_parts.append(f"User asked: {msg.content[:200]}")
            elif msg.role == "assistant":
                content = msg.content[:200] if msg.content else "(tool calls)"
                summary_parts.append(f"Assistant: {content}")
            elif msg.role == "tool":
                summary_parts.append(f"Tool result: {msg.content[:100]}")

        summary = "[Earlier conversation summary]\n" + "\n".join(summary_parts[-10:])
        summary_msg = Message(role="system", content=summary)

        return system_msgs + [summary_msg] + recent

    def build_messages(
        self,
        system_prompt: str,
        conversation: list[Message],
    ) -> list[Message]:
        """Build the final message list for LLM."""
        messages = [Message(role="system", content=system_prompt)]
        messages.extend(conversation)

        if self.should_compress(messages):
            # Compress conversation (keep system prompt)
            conv_only = conversation.copy()
            compressed = self.compress(conv_only)
            messages = [Message(role="system", content=system_prompt)] + compressed

        return messages
