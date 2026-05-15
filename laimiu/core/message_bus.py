"""Async publish/subscribe message bus for multi-agent communication."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("laimiu.message_bus")


@dataclass
class BusMessage:
    """A message on the inter-agent / agent→CLI bus."""

    source: str      # "brain" | "worker" | "system"
    task_id: str
    type: str        # mirrors OutputMessage.type
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """Simple async pub/sub message bus.

    Agents publish BusMessages to named channels.
    The CLI subscribes to an "output" channel to receive renderable messages.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[BusMessage]]] = {}

    async def publish(self, channel: str, msg: BusMessage) -> None:
        """Publish a message to all subscribers of *channel*."""
        for q in self._subscribers.get(channel, []):
            await q.put(msg)

    def subscribe(self, channel: str) -> asyncio.Queue[BusMessage]:
        """Subscribe to *channel*, returning a Queue that receives BusMessages."""
        q: asyncio.Queue[BusMessage] = asyncio.Queue()
        self._subscribers.setdefault(channel, []).append(q)
        return q

    async def get_next_output(
        self, queue: asyncio.Queue[BusMessage], timeout: float = 0.1
    ) -> BusMessage | None:
        """Try to get the next message with a short timeout."""
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
