"""Multi-agent orchestration: AgentWorker, AgentTask, AgentOrchestrator."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from laimiu.core.agent import AgentLoop
from laimiu.core.message_bus import BusMessage, MessageBus
from laimiu.core.messages import OutputMessage

logger = logging.getLogger("laimiu.agents")


class AgentRole(Enum):
    BRAIN = "brain"
    WORKER = "worker"


@dataclass
class AgentTask:
    """A task dispatched to an agent."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    role: AgentRole = AgentRole.BRAIN
    user_message: str = ""
    status: str = "pending"   # pending | running | done | error
    result: str | None = None


@dataclass
class AgentWorker:
    """Wraps an AgentLoop with role metadata."""

    name: str
    role: AgentRole
    agent: AgentLoop
    is_busy: bool = False
    current_task: AgentTask | None = None


class AgentOrchestrator:
    """Manages a pool of AgentWorkers and routes messages."""

    def __init__(self, message_bus: MessageBus) -> None:
        self.workers: dict[str, AgentWorker] = {}
        self.task_queue: asyncio.Queue[AgentTask] = asyncio.Queue()
        self.message_bus = message_bus
        self._output_queue = message_bus.subscribe("output")
        self._running = False

    # -- registration --------------------------------------------------

    def register_worker(self, worker: AgentWorker) -> None:
        self.workers[worker.name] = worker

    # -- routing -------------------------------------------------------

    def route_message(self, user_message: str) -> AgentTask:
        """Route a user message to the brain agent (non-blocking)."""
        task = AgentTask(
            role=AgentRole.BRAIN,
            user_message=user_message,
        )
        self.task_queue.put_nowait(task)
        return task

    def route_to_worker(self, user_message: str) -> AgentTask:
        """Explicitly route a task to the worker agent."""
        task = AgentTask(
            role=AgentRole.WORKER,
            user_message=user_message,
        )
        self.task_queue.put_nowait(task)
        return task

    # -- execution -----------------------------------------------------

    async def run_worker_loop(self) -> None:
        """Background loop: consume task queue and dispatch to workers."""
        self._running = True
        while self._running:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            worker = self._pick_worker(task.role)
            if worker is None:
                await self.message_bus.publish("output", BusMessage(
                    source="system",
                    task_id=task.id,
                    type="error",
                    content=f"No available worker for role {task.role.value}",
                ))
                continue

            asyncio.create_task(self._execute_task(worker, task))

    def _pick_worker(self, role: AgentRole) -> AgentWorker | None:
        """Pick an idle worker for the given role, or any idle worker."""
        # Prefer exact role match
        for w in self.workers.values():
            if w.role == role and not w.is_busy:
                return w
        # Fall back to any idle worker
        for w in self.workers.values():
            if not w.is_busy:
                return w
        # All busy — reject
        return None

    async def _execute_task(self, worker: AgentWorker, task: AgentTask) -> None:
        """Execute a task on a worker and publish output to the bus."""
        worker.is_busy = True
        worker.current_task = task
        task.status = "running"

        try:
            async for msg in worker.agent.run(task.user_message):
                bus_msg = BusMessage(
                    source=worker.name,
                    task_id=task.id,
                    type=msg.type,
                    content=msg.content,
                    metadata=msg.metadata,
                )
                await self.message_bus.publish("output", bus_msg)

            # Signal task completion so the CLI knows when to print separator
            await self.message_bus.publish("output", BusMessage(
                source=worker.name,
                task_id=task.id,
                type="task_done",
            ))
            task.status = "done"
        except Exception as e:
            logger.error(f"Worker {worker.name} task {task.id} failed: {e}")
            task.status = "error"
            task.result = str(e)
            await self.message_bus.publish("output", BusMessage(
                source=worker.name,
                task_id=task.id,
                type="error",
                content=str(e),
            ))
            await self.message_bus.publish("output", BusMessage(
                source=worker.name,
                task_id=task.id,
                type="task_done",
            ))
        finally:
            worker.is_busy = False
            worker.current_task = None

    # -- output --------------------------------------------------------

    def get_output_queue(self) -> asyncio.Queue[BusMessage]:
        """Return the output subscription queue for the CLI to consume."""
        return self._output_queue

    # -- status --------------------------------------------------------

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Return status of all workers."""
        status = {}
        for name, w in self.workers.items():
            status[name] = {
                "role": w.role.value,
                "busy": w.is_busy,
                "current_task": w.current_task.id if w.current_task else None,
            }
        return status

    # -- lifecycle -----------------------------------------------------

    def stop(self) -> None:
        self._running = False

    def start_sessions(self) -> None:
        for w in self.workers.values():
            w.agent.start_session()

    def end_sessions(self) -> None:
        for w in self.workers.values():
            w.agent.end_session()
