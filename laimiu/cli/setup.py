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
                "label": "GLM / ZhiPu (智谱)",
                "config": ProviderModelConfig(
                    base_url="https://open.bigmodel.cn/api/coding/paas/v4",
                    model="GLM-4-Plus",
                ),
                "needs_key": True,
                "key_prompt": "Enter your ZhiPu API Key: ",
                "key_env": "ZHIPU_API_KEY",
            },
            "3": {
                "name": "qwen",
                "label": "Qwen / Tongyi (通义千问)",
                "config": ProviderModelConfig(
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    model="qwen-plus",
                ),
                "needs_key": True,
                "key_prompt": "Enter your DashScope API Key: ",
                "key_env": "DASHSCOPE_API_KEY",
            },
            "4": {
                "name": "doubao",
                "label": "Doubao / Volcengine (豆包/火山引擎)",
                "config": ProviderModelConfig(
                    base_url="https://ark.cn-beijing.volces.com/api/v3",
                    model="doubao-pro-32k",
                ),
                "needs_key": True,
                "key_prompt": "Enter your Volcengine API Key: ",
                "key_env": "VOLCENGINE_API_KEY",
            },
            "5": {
                "name": "moonshot",
                "label": "Moonshot / Kimi (月之暗面)",
                "config": ProviderModelConfig(
                    base_url="https://api.moonshot.cn/v1",
                    model="moonshot-v1-8k",
                ),
                "needs_key": True,
                "key_prompt": "Enter your Moonshot API Key: ",
                "key_env": "MOONSHOT_API_KEY",
            },
            "6": {
                "name": "yi",
                "label": "Yi / ZeroOne (零一万物)",
                "config": ProviderModelConfig(
                    base_url="https://api.lingyiwanwu.com/v1",
                    model="yi-large",
                ),
                "needs_key": True,
                "key_prompt": "Enter your Yi API Key: ",
                "key_env": "YI_API_KEY",
            },
            "7": {
                "name": "baichuan",
                "label": "Baichuan (百川)",
                "config": ProviderModelConfig(
                    base_url="https://api.baichuan-ai.com/v1",
                    model="Baichuan4",
                ),
                "needs_key": True,
                "key_prompt": "Enter your Baichuan API Key: ",
                "key_env": "BAICHUAN_API_KEY",
            },
            "8": {
                "name": "minimax",
                "label": "MiniMax (海螺AI)",
                "config": ProviderModelConfig(
                    base_url="https://api.minimax.chat/v1",
                    model="MiniMax-Text-01",
                ),
                "needs_key": True,
                "key_prompt": "Enter your MiniMax API Key: ",
                "key_env": "MINIMAX_API_KEY",
            },
            "9": {
                "name": "siliconflow",
                "label": "SiliconFlow (硅基流动)",
                "config": ProviderModelConfig(
                    base_url="https://api.siliconflow.cn/v1",
                    model="Qwen/Qwen2.5-7B-Instruct",
                ),
                "needs_key": True,
                "key_prompt": "Enter your SiliconFlow API Key: ",
                "key_env": "SILICONFLOW_API_KEY",
            },
            "10": {
                "name": "openai",
                "label": "OpenAI",
                "config": ProviderModelConfig(
                    base_url="https://api.openai.com/v1",
                    model="gpt-4o-mini",
                ),
                "needs_key": True,
                "key_prompt": "Enter your OpenAI API Key: ",
                "key_env": "OPENAI_API_KEY",
            },
            "11": {
                "name": "ollama",
                "label": "Ollama (local, no API Key needed)",
                "config": ProviderModelConfig(
                    base_url="http://localhost:11434/v1",
                    model="llama3",
                    api_key="not-needed",
                ),
                "needs_key": False,
            },
            "12": {
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
        print("=" * 55)
        print("   Welcome to Laimiu!")
        print("   Self-Evolving AI Agent v0.2.0")
        print("=" * 55)
        print()
        print("Let's set up your configuration.")
        print("You can change these settings later by editing:")
        print(f"  {CONFIG_FILE}")
        print()

    def _ask_language(self) -> str:
        """Ask user for preferred language."""
        print("Select your preferred language / 选择语言:")
        print("  1. Chinese / 中文 (zh)")
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
        print("\nSelect your LLM provider / 选择大模型:")
        print("  ── 国内云服务 ──")
        print("  1.  DeepSeek")
        print("  2.  GLM / ZhiPu (智谱)")
        print("  3.  Qwen / Tongyi (通义千问)")
        print("  4.  Doubao / Volcengine (豆包/火山引擎)")
        print("  5.  Moonshot / Kimi (月之暗面)")
        print("  6.  Yi / ZeroOne (零一万物)")
        print("  7.  Baichuan (百川)")
        print("  8.  MiniMax (海螺AI)")
        print("  9.  SiliconFlow (硅基流动)")
        print("  ── 国际云服务 ──")
        print("  10. OpenAI")
        print("  ── 本地部署 ──")
        print("  11. Ollama (local, free, no API Key)")
        print("  ── 自定义 ──")
        print("  12. Other OpenAI-compatible")

        while True:
            choice = input("\nChoice [1]: ").strip() or "1"
            if choice in self._provider_choices:
                return choice, self._provider_choices[choice]
            print(f"Please enter a number between 1 and {len(self._provider_choices)}.")

    def _configure_provider(
        self, choice_key: str, provider_info: dict[str, Any]
    ) -> ProviderModelConfig:
        """Configure the selected provider."""
        if choice_key == "12":
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

        # Ask for model name (user can override default)
        print(f"\n  Default model: {config.model}")
        custom_model = input(f"  Model name (Enter to use default): ").strip()
        if custom_model:
            config.model = custom_model

        if provider_info.get("needs_key"):
            # Check env var first
            import os
            env_var = provider_info.get("key_env", "")
            env_key = os.environ.get(env_var, "") if env_var else ""
            if env_key:
                print(f"  Found {env_var} in environment, using it.")
                config.api_key = env_key
            else:
                key = input(f"  {provider_info.get('key_prompt', 'API Key: ')}").strip()
                config.api_key = key

        return config

    def _test_connection(self, provider_name: str, config: ProviderModelConfig) -> None:
        """Test the connection to the selected provider."""
        print(f"\n  Testing connection to {provider_name}...")

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

            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    provider.chat_complete([
                        Message(role="user", content="Hi, reply with just 'OK'."),
                    ])
                )
            finally:
                loop.close()

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
        print("=" * 55)
        print("   Setup complete! / 配置完成!")
        print("   Type anything to start chatting.")
        print("   Type /help for available commands.")
        print("=" * 55)
        print()
