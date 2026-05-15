"""Agent main loop - orchestrates conversation with all learning layers."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from laimiu.config.settings import LaimiuConfig
from laimiu.constants import SOUL_FILE
from laimiu.core.context import ContextManager
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

    async def run(self, user_message: str) -> AsyncIterator[str]:
        """Process a user message and stream the response.

        Yields text chunks as they arrive from the LLM.
        """
        # Add user message to conversation
        self.conversation.append(Message(role="user", content=user_message))

        # Build system prompt
        system_prompt = build_system_prompt(
            soul=self.soul,
            memory_index=self.memory.get_index(),
            tool_list=self.tools.list_tools(),
            user_prefs=self.memory.get_user_preferences(),
            provider_name=self._provider_name,
            model_name=self._model_name,
        )

        # Build messages within token budget
        messages = self.context_manager.build_messages(system_prompt, self.conversation)

        # Get LLM tools spec
        openai_tools = self.tools.get_openai_tools()

        # Agent loop
        self._tools_used_this_turn = []
        full_response = ""
        _chunk_count = 0

        while self.iterations < self.max_iterations:
            self.iterations += 1
            accumulated_content = ""
            accumulated_tool_calls: list[dict[str, Any]] = []
            accumulated_reasoning = ""

            # Stream from LLM
            provider = self.router.get_provider("chat")
            stream = provider.chat(messages, tools=openai_tools if openai_tools else None, stream=True)

            async for chunk in stream:
                if not isinstance(chunk, StreamChunk):
                    continue

                # Accumulate content
                if chunk.content:
                    accumulated_content += chunk.content
                    yield chunk.content

                    # Sentence-level repetition detection (every 30 chunks)
                    _chunk_count += 1
                    if _chunk_count % 30 == 0 and len(accumulated_content) > 200:
                        tail = accumulated_content[-500:]
                        # Check if last 40 chars appear 3+ times in recent text
                        pattern = tail[-40:]
                        if tail.count(pattern) >= 3:
                            logger.warning(
                                f"Repetition detected: '{pattern[:30]}...' "
                                f"appears {tail.count(pattern)} times, stopping"
                            )
                            break

                # Collect complete tool calls (delivered on finish_reason)
                if chunk.tool_calls:
                    accumulated_tool_calls = chunk.tool_calls

                # Accumulate reasoning content from thinking models
                if chunk.reasoning_content:
                    accumulated_reasoning += chunk.reasoning_content

                # Check if done
                if chunk.finish_reason in ("stop", "tool_calls"):
                    break

            # Handle tool calls
            if accumulated_tool_calls:
                # Parse tool calls from the complete data
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

                # Add assistant message with tool calls to conversation
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
                    content=accumulated_content,
                    tool_calls=assistant_tc_dicts,
                    reasoning_content=accumulated_reasoning or None,
                ))

                # Execute each tool call
                for tc in tool_calls:
                    result = await self.tool_executor.execute(tc)

                    # Reflect on result
                    reflection_result = self.reflection.evaluate(
                        tc.name, tc.arguments, result, user_message
                    )

                    # Track for procedural memory
                    if self.procedural_tracker:
                        self.procedural_tracker.record(
                            tc.name, tc.arguments, result, reflection_result
                        )

                    self._tools_used_this_turn.append({
                        "tool": tc.name,
                        "success": result.success,
                        "reflection_confidence": reflection_result.confidence,
                    })

                    # If failed and has alternative, retry
                    if reflection_result.should_retry and reflection_result.alternative_approach:
                        logger.info(f"Retrying with alternative: {reflection_result.alternative_approach}")

                    # Add tool result to conversation
                    self.conversation.append(Message(
                        role="tool",
                        content=result.to_text(),
                        tool_call_id=tc.id,
                        name=tc.name,
                    ))

                # Rebuild messages and continue loop
                messages = self.context_manager.build_messages(
                    system_prompt, self.conversation
                )
                accumulated_content = ""
                continue  # Next iteration to get LLM response to tool results

            else:
                # No tool calls - we're done
                full_response = accumulated_content
                # CRITICAL: add assistant response to conversation history
                self.conversation.append(Message(
                    role="assistant",
                    content=accumulated_content,
                    reasoning_content=accumulated_reasoning or None,
                ))
                break

        # Save turn to memory
        if full_response or self._tools_used_this_turn:
            self.memory.save_turn(
                user_message=user_message,
                assistant_response=full_response,
                tools_used=self._tools_used_this_turn,
            )

    async def run_complete(self, user_message: str) -> str:
        """Non-streaming version - returns full response."""
        chunks = []
        async for chunk in self.run(user_message):
            chunks.append(chunk)
        return "".join(chunks)

    def get_tools_used(self) -> list[dict[str, Any]]:
        """Get tools used in the current session."""
        return self.tool_executor.get_call_log()

    def get_stats(self) -> dict[str, Any]:
        """Get current session stats."""
        return {
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "conversation_turns": len([m for m in self.conversation if m.role == "user"]),
            "tools_used": len(self.tool_executor.get_call_log()),
        }
