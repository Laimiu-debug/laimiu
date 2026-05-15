"""Model router - selects provider for different tasks."""

from __future__ import annotations

import logging
from typing import Any

from laimiu.config.settings import LaimiuConfig, ProviderModelConfig
from laimiu.providers.base import LLMResponse, Message, ProviderProfile
from laimiu.providers.openai_compat import OpenAICompatProvider

logger = logging.getLogger("laimiu.providers.router")


class ProviderRouter:
    """Routes requests to appropriate LLM providers."""

    def __init__(self, config: LaimiuConfig):
        self.config = config
        self._providers: dict[str, OpenAICompatProvider] = {}
        self._init_providers()

    def _init_providers(self) -> None:
        """Initialize providers from config."""
        for name, model_cfg in self.config.provider.models.items():
            profile = ProviderProfile(
                name=name,
                base_url=model_cfg.base_url,
                model=model_cfg.model,
                api_key=model_cfg.api_key,
                temperature=model_cfg.temperature,
                max_tokens=model_cfg.max_tokens,
            )
            self._providers[name] = OpenAICompatProvider(profile)

    def get_provider(self, task: str = "chat") -> OpenAICompatProvider:
        """Get provider for a given task.

        Args:
            task: "chat" for main conversation, "dream" for dream engine,
                  "cheap" for lightweight tasks.

        Returns:
            The appropriate provider.
        """
        if task == "dream":
            dream_model = self.config.dream.model
            if dream_model in self._providers:
                return self._providers[dream_model]

        if task == "cheap":
            # Use Ollama if available for cheap tasks
            if "ollama" in self._providers:
                return self._providers["ollama"]

        # Default: use the configured default provider
        default_name = self.config.provider.default
        if default_name in self._providers:
            return self._providers[default_name]

        # Fallback to first available
        if self._providers:
            return next(iter(self._providers.values()))

        raise ValueError("No providers configured")

    def get_provider_by_name(self, name: str) -> OpenAICompatProvider | None:
        """Get a specific provider by name."""
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        """List all available provider names."""
        return list(self._providers.keys())

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        task: str = "chat",
    ):
        """Chat using the appropriate provider."""
        provider = self.get_provider(task)
        return provider.chat(messages, tools=tools, stream=stream)

    async def chat_complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        task: str = "chat",
    ) -> LLMResponse:
        """Non-streaming chat using the appropriate provider."""
        provider = self.get_provider(task)
        return await provider.chat_complete(messages, tools=tools)
