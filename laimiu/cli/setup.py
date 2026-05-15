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

# Provider category metadata for colored display
_PROVIDER_CATEGORIES = {
    "1": ("cn", "\U0001f525"),   # DeepSeek
    "2": ("cn", "\U0001f4a1"),   # GLM
    "3": ("cn", "\u2600"),       # Qwen
    "4": ("cn", "\U0001f4e6"),   # Doubao
    "5": ("cn", "\U0001f319"),   # Moonshot
    "6": ("cn", "\U0001f30a"),   # Yi
    "7": ("cn", "\U0001f432"),   # Baichuan
    "8": ("cn", "\U0001f4ab"),   # MiniMax
    "9": ("cn", "\U0001f4bb"),   # SiliconFlow
    "10": ("intl", "\U0001f310"),  # OpenAI
    "11": ("local", "\U0001f4be"), # Ollama
    "12": ("custom", "\u2699"),    # Custom
}


class SetupWizard:
    """Interactive first-run configuration wizard with Rich UI."""

    def __init__(self):
        try:
            from rich.console import Console
            self.console = Console()
            self._use_rich = True
        except ImportError:
            self.console = None
            self._use_rich = False

        self._provider_choices: dict[str, dict[str, Any]] = {
            "1": {
                "name": "deepseek",
                "label": "DeepSeek",
                "config": ProviderModelConfig(
                    base_url="https://api.deepseek.com",
                    model="deepseek-v4-pro",
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
                    model="glm-4.7-flashx",
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
                    model="doubao-1.5-pro",
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
                    model="Qwen/Qwen3-8B",
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
                    model="gpt-4.1-mini",
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
                    model="llama3.1",
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

    # ── Greeting ──────────────────────────────────────────────

    def _greet(self) -> None:
        if self._use_rich:
            from rich.panel import Panel
            from rich.rule import Rule
            from rich.text import Text

            self.console.print()
            self.console.print(Rule(style="cyan"))
            self.console.print(Panel(
                Text("Welcome to Laimiu!\nSelf-Evolving AI Agent v0.2.0", justify="center"),
                border_style="cyan",
                padding=(1, 2),
            ))
            self.console.print(Rule(style="cyan"))
            self.console.print()
            self.console.print("Let's set up your configuration.")
            self.console.print(f"You can change these settings later by editing: [dim]{CONFIG_FILE}[/dim]")
            self.console.print()
        else:
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

    # ── Language selection ────────────────────────────────────

    def _ask_language(self) -> str:
        """Ask user for preferred language."""
        if self._use_rich:
            from rich.table import Table

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="bold cyan")
            table.add_column()
            table.add_row("1", "Chinese / 中文 (zh)")
            table.add_row("2", "English (en)")

            self.console.print()
            self.console.print("Select your preferred language / 选择语言:")
            self.console.print(table)
        else:
            print("Select your preferred language / 选择语言:")
            print("  1. Chinese / 中文 (zh)")
            print("  2. English (en)")

        while True:
            choice = self._prompt_input("\nChoice [1]: ", default="1")
            if choice in ("1", "zh", "chinese"):
                return "zh"
            elif choice in ("2", "en", "english"):
                return "en"
            self._print_error("Please enter 1 or 2.")

    # ── Provider selection ────────────────────────────────────

    def _ask_provider(self) -> tuple[str, dict[str, Any]]:
        """Ask user which LLM provider to use."""
        if self._use_rich:
            self._ask_provider_rich()
        else:
            self._ask_provider_plain()

        while True:
            choice = self._prompt_input("\nChoice [1]: ", default="1")
            if choice in self._provider_choices:
                return choice, self._provider_choices[choice]
            self._print_error(
                f"Please enter a number between 1 and {len(self._provider_choices)}."
            )

    def _ask_provider_rich(self) -> None:
        """Render provider selection table with Rich."""
        from rich.table import Table
        from rich.text import Text

        table = Table(
            title="Select your LLM provider / 选择大模型",
            show_header=False,
            box=None,
            padding=(0, 1),
        )
        table.add_column(min_width=3)
        table.add_column()
        table.add_column(style="dim")

        sections = [
            ("── 国内云服务 ──", ["1", "2", "3", "4", "5", "6", "7", "8", "9"]),
            ("── 国际云服务 ──", ["10"]),
            ("── 本地部署 ──", ["11"]),
            ("── 自定义 ──", ["12"]),
        ]

        first = True
        for section_title, keys in sections:
            if not first:
                table.add_row("", Text(section_title, style="dim cyan"))
            else:
                table.add_row("", Text(section_title, style="dim cyan"))
            first = False
            for key in keys:
                info = self._provider_choices[key]
                icon = _PROVIDER_CATEGORIES.get(key, ("", "\u2022"))[1]
                model = info["config"].model if info["config"] else ""
                table.add_row(
                    Text(key, style="bold green"),
                    Text(f"{icon} {info['label']}", style="white"),
                    Text(model, style="dim"),
                )

        self.console.print()
        self.console.print(table)

    def _ask_provider_plain(self) -> None:
        """Render provider selection as plain text."""
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

    # ── Provider configuration ────────────────────────────────

    def _configure_provider(
        self, choice_key: str, provider_info: dict[str, Any]
    ) -> ProviderModelConfig:
        """Configure the selected provider."""
        if choice_key == "12":
            # Custom provider
            self._print_section("Custom Provider Configuration")
            base_url = self._prompt_input("Base URL: ").strip()
            model = self._prompt_input("Model name: ").strip()
            api_key = self._prompt_input("API Key (press Enter if none): ").strip() or "not-needed"
            return ProviderModelConfig(
                base_url=base_url,
                model=model,
                api_key=api_key,
            )

        config = provider_info["config"]

        # Ask for model name (user can override default)
        self._print_section(provider_info["label"])
        if self._use_rich:
            self.console.print(f"  Default model: [green]{config.model}[/green]")
        else:
            print(f"  Default model: {config.model}")

        custom_model = self._prompt_input("  Model name (Enter to use default): ").strip()
        if custom_model:
            config.model = custom_model

        if provider_info.get("needs_key"):
            # Check env var first
            import os
            env_var = provider_info.get("key_env", "")
            env_key = os.environ.get(env_var, "") if env_var else ""
            if env_key:
                if self._use_rich:
                    self.console.print(f"  [green]Found {env_var} in environment, using it.[/green]")
                else:
                    print(f"  Found {env_var} in environment, using it.")
                config.api_key = env_key
            else:
                key = self._prompt_input(
                    f"  {provider_info.get('key_prompt', 'API Key: ')}"
                ).strip()
                config.api_key = key

        return config

    # ── Connection test ───────────────────────────────────────

    def _test_connection(self, provider_name: str, config: ProviderModelConfig) -> None:
        """Test the connection to the selected provider."""
        if self._use_rich:
            from rich.live import Live
            from rich.spinner import Spinner
            from rich.text import Text

            self.console.print()
            with Live(
                Spinner("dots", Text(f"  Testing connection to {provider_name}...", style="cyan")),
                console=self.console,
                refresh_per_second=10,
                transient=True,
            ):
                result = self._do_connection_test(provider_name, config)

            if result == "ok":
                self.console.print(f"  [green]\u2713 Connection successful! ({provider_name}/{config.model})[/green]")
            elif result == "empty":
                self.console.print("  [yellow]Warning: Got empty response, but connection was made.[/yellow]")
            elif result == "no_openai":
                self.console.print("  [dim]Skipping connection test (openai package not installed).[/dim]")
            else:
                self.console.print(f"  [red]Warning: Connection test failed: {result}[/red]")
                self.console.print("  [dim]You can still proceed. Check your API Key and network settings.[/dim]")
        else:
            print(f"\n  Testing connection to {provider_name}...")
            result = self._do_connection_test(provider_name, config)
            if result == "ok":
                print(f"  Connection successful! ({provider_name}/{config.model})")
            elif result == "empty":
                print("  Warning: Got empty response, but connection was made.")
            elif result == "no_openai":
                print("  Skipping connection test (openai package not installed).")
            else:
                print(f"  Warning: Connection test failed: {result}")
                print("  You can still proceed. Check your API Key and network settings.")

    def _do_connection_test(self, provider_name: str, config: ProviderModelConfig) -> str:
        """Run connection test. Returns status string."""
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
                return "ok"
            return "empty"
        except ImportError:
            return "no_openai"
        except Exception as e:
            return str(e)

    # ── Config build / save ───────────────────────────────────

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
        if self._use_rich:
            self.console.print(f"\n  [green]\u2713 Configuration saved to[/green] [dim]{CONFIG_FILE}[/dim]")
        else:
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
        if self._use_rich:
            self.console.print(f"  [green]\u2713 Soul file created at[/green] [dim]{SOUL_FILE}[/dim]")
        else:
            print(f"  Soul file created at {SOUL_FILE}")

    # ── Done ──────────────────────────────────────────────────

    def _done(self) -> None:
        if self._use_rich:
            from rich.panel import Panel
            from rich.rule import Rule

            self.console.print()
            self.console.print(Rule(style="green"))
            self.console.print(Panel(
                "[bold green]Setup complete! / 配置完成![/bold green]\n\n"
                "Type anything to start chatting.\n"
                "Type /help for available commands.",
                border_style="green",
                padding=(1, 2),
            ))
            self.console.print(Rule(style="green"))
            self.console.print()
        else:
            print()
            print("=" * 55)
            print("   Setup complete! / 配置完成!")
            print("   Type anything to start chatting.")
            print("   Type /help for available commands.")
            print("=" * 55)
            print()

    # ── Helpers ───────────────────────────────────────────────

    def _prompt_input(self, prompt: str, default: str = "") -> str:
        """Read input with optional Rich styling."""
        raw = input(prompt).strip()
        return raw or default

    def _print_error(self, message: str) -> None:
        """Print error with styling."""
        if self._use_rich:
            self.console.print(f"  [red]{message}[/red]")
        else:
            print(f"  {message}")

    def _print_section(self, title: str) -> None:
        """Print a section header."""
        if self._use_rich:
            self.console.print()
            self.console.print(f"  [bold cyan]{title}[/bold cyan]")
        else:
            print(f"\n--- {title} ---")
