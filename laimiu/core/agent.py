"""Agent main loop - orchestrates conversation with all learning layers."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

from laimiu.config.settings import LaimiuConfig
from laimiu.constants import SOUL_FILE
from laimiu.core.context import ContextManager
from laimiu.core.messages import OutputMessage
from laimiu.core.prompt import build_system_prompt, load_soul
from laimiu.core.reflection import Reflection, ReflectionResult
from laimiu.core.tool_executor import ToolExecutor
from laimiu.memory.manager import MemoryManager
from laimiu.providers.base import LLMResponse, Message, StreamChunk, ToolCall
from laimiu.providers.router import ProviderRouter
from laimiu.tools.registry import ToolRegistry

logger = logging.getLogger("laimiu.agent")


class AgentLoop:
    """Main agent loop integrating all three learning layers.

    Flow:
    1. Receive user message
    2. Assemble context (SOUL + memory index + tool list)
    3. Call LLM (streaming)
    4. If tool calls: execute -> reflect -> possibly retry
    5. Stream response to user
    6. Save turn to memory
    7. Check dream trigger
    """

    def __init__(
        self,
        config: LaimiuConfig,
        memory: MemoryManager,
        tool_registry: ToolRegistry,
        router: ProviderRouter,
        reflection: Reflection | None = None,
        procedural_tracker: Any = None,
    ):
        self.config = config
        self.memory = memory
        self.tools = tool_registry
        self.router = router
        self.reflection = reflection or Reflection()
        self.procedural_tracker = procedural_tracker

        self.tool_executor = ToolExecutor(tool_registry)
        self.context_manager = ContextManager()

        self.conversation: list[Message] = []
        self.iterations = 0
        self.max_iterations = config.agent.max_turns

        # Load soul
        self.soul = load_soul()

        # Provider/model info for system prompt identity
        self._provider_name = config.provider.default
        self._model_name = ""
        if config.provider.models.get(self._provider_name):
            self._model_name = config.provider.models[self._provider_name].model

        # Session tracking
        self._session_id: str | None = None
        self._tools_used_this_turn: list[dict[str, Any]] = []

    def start_session(self) -> str:
        """Start a new conversation session."""
        self._session_id = self.memory.start_session()
        self.conversation.clear()
        self.iterations = 0
        return self._session_id

    def end_session(self) -> None:
        """End the current session."""
        self.memory.end_session()
        self._session_id = None

    async def run(self, user_message: str) -> AsyncIterator[OutputMessage]:
        """Process a user message and stream structured output messages."""
        self.conversation.append(Message(role="user", content=user_message))

        system_prompt = build_system_prompt(
            soul=self.soul,
            memory_index=self.memory.get_index(),
            tool_list=self.tools.list_tools(),
            user_prefs=self.memory.get_user_preferences(),
            provider_name=self._provider_name,
            model_name=self._model_name,
        )

        messages = self.context_manager.build_messages(system_prompt, self.conversation)
        openai_tools = self.tools.get_openai_tools()

        self._tools_used_this_turn = []
        full_response = ""
        _chunk_count = 0

        while self.iterations < self.max_iterations:
            self.iterations += 1
            accumulated_content = ""
            accumulated_tool_calls: list[dict[str, Any]] = []
            accumulated_reasoning = ""

            provider = self.router.get_provider("chat")
            stream = provider.chat(messages, tools=openai_tools if openai_tools else None, stream=True)

            async for chunk in stream:
                if not isinstance(chunk, StreamChunk):
                    continue

                if chunk.reasoning_content:
                    accumulated_reasoning += chunk.reasoning_content
                    yield OutputMessage.thinking(chunk.reasoning_content)

                if chunk.content:
                    accumulated_content += chunk.content
                    yield OutputMessage.content_chunk(chunk.content)

                    _chunk_count += 1
                    if _chunk_count % 30 == 0 and len(accumulated_content) > 200:
                        tail = accumulated_content[-500:]
                        pattern = tail[-40:]
                        if tail.count(pattern) >= 3:
                            logger.warning(
                                f"Repetition detected: '{pattern[:30]}...' "
                                f"appears {tail.count(pattern)} times, stopping"
                            )
                            break

                if chunk.tool_calls:
                    accumulated_tool_calls = chunk.tool_calls

                if chunk.finish_reason in ("stop", "tool_calls"):
                    break

            # Handle tool calls
            if accumulated_tool_calls:
                tool_calls = []
                for tc_data in accumulated_tool_calls:
                    fn = tc_data.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}")) if fn.get("arguments") else {}
                    except json.JSONDecodeError:
                        args = {"raw": fn.get("arguments", "")}

                    tool_calls.append(ToolCall(
                        id=tc_data.get("id", f"call_{len(tool_calls)}"),
                        name=fn.get("name", ""),
                        arguments=args,
                    ))

                assistant_tc_dicts = []
                for tc in tool_calls:
                    assistant_tc_dicts.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    })
                self.conversation.append(Message(
                    role="assistant",
                    content=accumulated_content or " ",
                    tool_calls=assistant_tc_dicts,
                    reasoning_content=accumulated_reasoning or None,
                ))

                for tc in tool_calls:
                    yield OutputMessage.tool_call_start(tc.name, tc.arguments)

                    t0 = time.perf_counter()
                    result = await self.tool_executor.execute(tc)
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    yield OutputMessage.tool_call_end(tc.name, elapsed_ms, result.success)

                    reflection_result = self.reflection.evaluate(
                        tc.name, tc.arguments, result, user_message
                    )

                    if self.procedural_tracker:
                        self.procedural_tracker.record(
                            tc.name, tc.arguments, result, reflection_result
                        )

                    self._tools_used_this_turn.append({
                        "tool": tc.name,
                        "success": result.success,
                        "reflection_confidence": reflection_result.confidence,
                    })

                    if reflection_result.should_retry and reflection_result.alternative_approach:
                        logger.info(f"Retrying with alternative: {reflection_result.alternative_approach}")

                    self.conversation.append(Message(
                        role="tool",
                        content=result.to_text(),
                        tool_call_id=tc.id,
                        name=tc.name,
                    ))

                messages = self.context_manager.build_messages(system_prompt, self.conversation)
                accumulated_content = ""
                continue

            else:
                full_response = accumulated_content
                self.conversation.append(Message(
                    role="assistant",
                    content=accumulated_content or " ",
                    reasoning_content=accumulated_reasoning or None,
                ))
                break

        if full_response or self._tools_used_this_turn:
            self.memory.save_turn(
                user_message=user_message,
                assistant_response=full_response,
                tools_used=self._tools_used_this_turn,
            )

    async def run_complete(self, user_message: str) -> str:
        """Non-streaming version - returns full response."""
        chunks = []
        async for msg in self.run(user_message):
            if msg.type in ("content", "thinking"):
                chunks.append(msg.content)
        return "".join(chunks)

    def get_tools_used(self) -> list[dict[str, Any]]:
        return self.tool_executor.get_call_log()

    def get_stats(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "conversation_turns": len([m for m in self.conversation if m.role == "user"]),
            "tools_used": len(self.tool_executor.get_call_log()),
        }
