"""First-run setup wizard for Laimiu configuration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from laimiu.config.settings import (
    LaimiuConfig,
    ProviderConfig,
    ProviderModelConfig,
)
from laimiu.constants import (
    CONFIG_FILE,
    LAIMIU_HOME,
    SOUL_FILE,
    ensure_dirs,
)
from laimiu.utils.io import atomic_write

logger = logging.getLogger("laimiu.cli.setup")


class SetupWizard:
    """Interactive first-run configuration wizard."""

    def __init__(self):
        self._provider_choices: dict[str, dict[str, Any]] = {
            "1": {
                "name": "deepseek",
                "label": "DeepSeek",
                "config": ProviderModelConfig(
                    base_url="https://api.deepseek.com",
                    model="deepseek-chat",
                ),
                "needs_key": True,
                "key_prompt": "Enter your DeepSeek API Key: ",
                "key_env": "DEEPSEEK_API_KEY",
            },
            "2": {
                "name": "glm",
                "label": "GLM/ZhiPu",
                "config": ProviderModelConfig(
                    base_url="https://open.bigmodel.cn/api/coding/paas/v4",
                    model="GLM-4-Plus",
                ),
                "needs_key": True,
                "key_prompt": "Enter your GLM API Key: ",
                "key_env": "OPENAI_API_KEY",
            },
            "3": {
                "name": "ollama",
                "label": "Ollama (local)",
                "config": ProviderModelConfig(
                    base_url="http://localhost:11434/v1",
                    model="llama3",
                    api_key="not-needed",
                ),
                "needs_key": False,
            },
            "4": {
                "name": "custom",
                "label": "Other OpenAI-compatible",
                "config": None,
                "needs_key": True,
            },
        }

    def run(self) -> LaimiuConfig:
        """Run the setup wizard. Returns a fully configured LaimiuConfig."""
        self._greet()
        language = self._ask_language()
        provider_key, provider_info = self._ask_provider()
        model_config = self._configure_provider(provider_key, provider_info)
        self._test_connection(provider_info["name"], model_config)
        config = self._build_config(language, provider_info["name"], model_config)
        self._save(config)
        self._create_soul(language)
        self._done()
        return config

    def _greet(self) -> None:
        print()
        print("=" * 50)
        print("  Welcome to Laimiu!")
        print("  Self-Evolving AI Agent")
        print("=" * 50)
        print()
        print("Let's set up your configuration.")
        print("You can change these settings later by editing:")
        print(f"  {CONFIG_FILE}")
        print()

    def _ask_language(self) -> str:
        """Ask user for preferred language."""
        print("Select your preferred language:")
        print("  1. Chinese (zh)")
        print("  2. English (en)")
        while True:
            choice = input("\nChoice [1]: ").strip() or "1"
            if choice in ("1", "zh", "chinese"):
                return "zh"
            elif choice in ("2", "en", "english"):
                return "en"
            print("Please enter 1 or 2.")

    def _ask_provider(self) -> tuple[str, dict[str, Any]]:
        """Ask user which LLM provider to use."""
        print("\nSelect your LLM provider:")
        print("  1. DeepSeek (cloud, requires API Key)")
        print("  2. GLM/ZhiPu (cloud, requires API Key)")
        print("  3. Ollama (local, no API Key needed)")
        print("  4. Other OpenAI-compatible")

        while True:
            choice = input("\nChoice [1]: ").strip() or "1"
            if choice in self._provider_choices:
                return choice, self._provider_choices[choice]
            print("Please enter 1, 2, 3, or 4.")

    def _configure_provider(
        self, choice_key: str, provider_info: dict[str, Any]
    ) -> ProviderModelConfig:
        """Configure the selected provider."""
        if choice_key == "4":
            # Custom provider
            print("\n--- Custom Provider Configuration ---")
            base_url = input("Base URL: ").strip()
            model = input("Model name: ").strip()
            api_key = input("API Key (press Enter if none): ").strip() or "not-needed"
            return ProviderModelConfig(
                base_url=base_url,
                model=model,
                api_key=api_key,
            )

        config = provider_info["config"]

        if provider_info.get("needs_key"):
            # Check env var first
            import os
            env_var = provider_info.get("key_env", "")
            env_key = os.environ.get(env_var, "") if env_var else ""
            if env_key:
                print(f"\n  Found {env_var} in environment, using it.")
                config.api_key = env_key
            else:
                key = input(f"\n{provider_info.get('key_prompt', 'API Key: ')}").strip()
                config.api_key = key

        return config

    def _test_connection(self, provider_name: str, config: ProviderModelConfig) -> None:
        """Test the connection to the selected provider."""
        print(f"\nTesting connection to {provider_name}...")

        try:
            from laimiu.providers.openai_compat import OpenAICompatProvider
            from laimiu.providers.base import ProviderProfile, Message

            profile = ProviderProfile(
                name=provider_name,
                base_url=config.base_url,
                model=config.model,
                api_key=config.api_key,
            )
            provider = OpenAICompatProvider(profile)

            result = asyncio.get_event_loop().run_until_complete(
                provider.chat_complete([
                    Message(role="user", content="Hi, reply with just 'OK'."),
                ])
            )

            if result and result.content:
                print(f"  Connection successful! ({provider_name}/{config.model})")
            else:
                print("  Warning: Got empty response, but connection was made.")
        except ImportError:
            print("  Skipping connection test (openai package not installed).")
        except Exception as e:
            print(f"  Warning: Connection test failed: {e}")
            print("  You can still proceed. Check your API Key and network settings.")

    def _build_config(
        self, language: str, provider_name: str, model_config: ProviderModelConfig,
    ) -> LaimiuConfig:
        """Build the full LaimiuConfig."""
        config = LaimiuConfig()
        config.provider = ProviderConfig(
            default=provider_name,
            models={provider_name: model_config},
        )
        return config

    def _save(self, config: LaimiuConfig) -> None:
        """Save configuration to disk."""
        ensure_dirs()
        config.save(CONFIG_FILE)
        print(f"\n  Configuration saved to {CONFIG_FILE}")

    def _create_soul(self, language: str) -> None:
        """Create SOUL.md with language-appropriate defaults."""
        if language == "zh":
            content = """# Laimiu

你是 Laimiu，一个能够学习和进化的个人 AI 助手。

## 性格
- 直接、有帮助、积极主动
- 记住用户的偏好并加以应用
- 不确定时，提问而非猜测
- 善用工具来完成任务

## 行为
- 使用简体中文回复
- 简洁而全面
- 需要时使用 memory_recall 查看历史对话
- 持续学习和改进
"""
        else:
            content = """# Laimiu

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
- Track your own learning and improvement
"""
        atomic_write(SOUL_FILE, content)
        print(f"  Soul file created at {SOUL_FILE}")

    def _done(self) -> None:
        print()
        print("=" * 50)
        print("  Setup complete!")
        print("  Type anything to start chatting.")
        print("  Type /help for available commands.")
        print("=" * 50)
        print()
