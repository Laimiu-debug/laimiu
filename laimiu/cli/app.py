"""CLI application - Rich + prompt_toolkit interactive chat."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from typing import Any

from laimiu import __version__
from laimiu.config.settings import LaimiuConfig
from laimiu.constants import (
    CONFIG_FILE,
    LAIMIU_HOME,
    SOUL_FILE,
    ensure_dirs,
)
from laimiu.core.agent import AgentLoop
from laimiu.core.reflection import Reflection
from laimiu.dream.engine import DreamEngine
from laimiu.memory.manager import MemoryManager
from laimiu.procedural.engine import ProceduralEngine
from laimiu.procedural.tracker import PatternTracker
from laimiu.providers.router import ProviderRouter
from laimiu.safety.guardian import Guardian
from laimiu.tools.registry import ToolRegistry, discover_builtin_tools, discover_generated_tools

logger = logging.getLogger("laimiu.cli")


def _is_first_run() -> bool:
    """Check if this is the first time Laimiu is being run."""
    return not LAIMIU_HOME.exists() or not any(LAIMIU_HOME.iterdir())


def _init_runtime() -> None:
    """Initialize the runtime environment."""
    ensure_dirs()

    # Create default SOUL.md if not exists
    if not SOUL_FILE.exists():
        from laimiu.utils.io import atomic_write
        atomic_write(SOUL_FILE, """# Laimiu

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
""")

    # Create default config if not exists
    if not CONFIG_FILE.exists():
        config = LaimiuConfig()
        # Set up default providers from env vars
        import os

        models = {}
        if os.environ.get("DEEPSEEK_API_KEY"):
            from laimiu.config.settings import ProviderModelConfig
            models["deepseek"] = ProviderModelConfig(
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
                api_key=os.environ["DEEPSEEK_API_KEY"],
            )
            config.provider.default = "deepseek"

        if os.environ.get("OPENAI_API_KEY"):
            from laimiu.config.settings import ProviderModelConfig
            models["glm"] = ProviderModelConfig(
                base_url="https://open.bigmodel.cn/api/coding/paas/v4",
                model="GLM-4-Plus",
                api_key=os.environ["OPENAI_API_KEY"],
            )
            if not models.get("deepseek"):
                config.provider.default = "glm"

        if not models:
            # Default to Ollama
            from laimiu.config.settings import ProviderModelConfig
            models["ollama"] = ProviderModelConfig(
                base_url="http://localhost:11434/v1",
                model="llama3",
                api_key="not-needed",
            )
            config.provider.default = "ollama"

        config.provider.models = models
        config.save(CONFIG_FILE)


def _print_banner(config: LaimiuConfig, agent: AgentLoop, memory: MemoryManager) -> None:
    """Print startup banner."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        # Get provider info
        provider_name = config.provider.default
        model_name = "unknown"
        if config.provider.models.get(provider_name):
            model_name = config.provider.models[provider_name].model

        # Get tool counts
        builtin_count = len([t for t in agent.tools.list_tools()])
        stats = memory.get_stats()

        banner_text = Text()
        banner_text.append(f"Laimiu v{__version__}\n", style="bold cyan")
        banner_text.append(f"Model: ", style="dim")
        banner_text.append(f"{provider_name}/{model_name}\n", style="green")
        banner_text.append(f"Tools: ", style="dim")
        banner_text.append(f"{builtin_count} registered", style="yellow")
        banner_text.append(f" | ", style="dim")
        banner_text.append(f"Memory: ", style="dim")
        banner_text.append(f"{stats.get('vector_notes', 0)} notes", style="magenta")

        console.print(Panel(banner_text, border_style="cyan", padding=(1, 2)))
    except ImportError:
        print(f"Laimiu v{__version__}")
        print(f"Model: {config.provider.default}")
        print(f"Tools: {len(agent.tools.list_tools())}")


async def _run_chat(config: LaimiuConfig) -> None:
    """Main chat loop."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.live import Live
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

        console = Console()
        use_rich = True
    except ImportError:
        console = None
        use_rich = False

    # Initialize components
    router = ProviderRouter(config)
    memory = MemoryManager(config)
    registry = ToolRegistry()

    # Discover tools
    discover_builtin_tools(registry)
    generated_count = discover_generated_tools(registry)

    # Create memory recall tool with manager reference
    from laimiu.tools.builtin.memory import MemoryRecallTool
    recall_tool = MemoryRecallTool(memory_manager=memory)
    registry.register(recall_tool)

    # Procedural memory - use strength-based threshold
    tracker = PatternTracker(config.procedural.extract_strength)
    procedural = ProceduralEngine(config, tracker, registry, router)

    # Reflection
    reflection = Reflection()

    # Agent
    agent = AgentLoop(
        config=config,
        memory=memory,
        tool_registry=registry,
        router=router,
        reflection=reflection,
        procedural_tracker=tracker,
    )

    # Dream engine
    dream = DreamEngine(config, memory, procedural, router)

    # Guardian for health checks
    guardian = Guardian()

    # Print banner
    if use_rich:
        _print_banner(config, agent, memory)
    else:
        print(f"Laimiu v{__version__} | Model: {config.provider.default}")

    if generated_count > 0:
        print(f"  Loaded {generated_count} learned tools")

    # Start session
    session_id = agent.start_session()

    # Prompt session
    history_file = LAIMIU_HOME / "chat_history"
    if use_rich:
        session = PromptSession(history=FileHistory(str(history_file)))
    else:
        session = None

    while True:
        try:
            # Get user input
            if use_rich:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: session.prompt("Laimiu> ", auto_suggest=AutoSuggestFromHistory())
                )
            else:
                user_input = input("Laimiu> ")

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                should_continue = await _handle_command(
                    user_input, config, agent, memory, dream, procedural,
                    guardian, console, use_rich
                )
                if not should_continue:
                    break
                continue

            # Regular chat
            if use_rich:
                from rich.text import Text

                with Live(console=console, refresh_per_second=10, vertical_overflow="visible") as live:
                    # Show thinking indicator
                    live.update(Text("Thinking...", style="dim italic"))
                    full_response = ""
                    async for chunk in agent.run(user_input):
                        full_response += chunk
                        live.update(Markdown(full_response))
                    # Final render
                    if full_response:
                        live.update(Markdown(full_response))
                    else:
                        live.update(Text("(no response)", style="dim"))
                console.print()
            else:
                print("Thinking...", flush=True)
                async for chunk in agent.run(user_input):
                    print(chunk, end="", flush=True)
                print()

        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            break
        except Exception as e:
            logger.error(f"Chat error: {e}")
            if use_rich:
                console.print(f"[red]Error: {e}[/red]")
            else:
                print(f"Error: {e}")

    # End session
    agent.end_session()
    dream.increment_sessions()

    # Check if dream should run
    if dream.should_dream():
        if use_rich:
            console.print("[dim]Running dream cycle...[/dim]")
        dream_results = await dream.dream()
        if use_rich and not dream_results.get("skipped"):
            console.print("[dim]Dream cycle completed[/dim]")


async def _handle_command(
    cmd: str,
    config: LaimiuConfig,
    agent: AgentLoop,
    memory: MemoryManager,
    dream: DreamEngine,
    procedural: ProceduralEngine,
    guardian: Guardian,
    console: Any,
    use_rich: bool,
) -> bool:
    """Handle slash commands. Returns False if should exit."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command in ("/quit", "/exit", "/q"):
        return False

    elif command == "/model":
        providers = agent.router.list_providers()
        if args:
            # Switch model
            provider = agent.router.get_provider_by_name(args.strip())
            if provider:
                config.provider.default = args.strip()
                if use_rich:
                    console.print(f"[green]Switched to {args.strip()}[/green]")
            else:
                if use_rich:
                    console.print(f"[red]Unknown provider. Available: {', '.join(providers)}[/red]")
        else:
            current = config.provider.default
            if use_rich:
                console.print(f"Current: [green]{current}[/green]")
                console.print(f"Available: {', '.join(providers)}")

    elif command == "/dream":
        if use_rich:
            console.print("[cyan]Running dream cycle...[/cyan]")
        results = await dream.dream()
        if use_rich:
            phases = results.get("phases", {})
            console.print(f"[green]Dream complete[/green] ({results.get('duration_seconds', 0):.1f}s)")
            if "consolidate" in phases:
                c = phases["consolidate"]
                console.print(f"  Notes created: {c.get('notes_created', 0)}")
            if "extract" in phases:
                e = phases["extract"]
                tools = e.get("tools_generated", [])
                console.print(f"  Tools generated: {len(tools)}")
                for t in tools:
                    console.print(f"    - {t}")

    elif command == "/memory":
        stats = memory.get_stats()
        index = memory.get_index()
        if use_rich:
            from rich.table import Table
            from rich.markdown import Markdown

            table = Table(title="Memory Stats")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            for k, v in stats.items():
                table.add_row(str(k), str(v))
            console.print(table)
            console.print(Markdown(index[:1000]))

    elif command == "/tools":
        tools = agent.tools.list_tools()
        if use_rich:
            from rich.table import Table
            table = Table(title="Registered Tools")
            table.add_column("Name", style="cyan")
            table.add_column("Description")
            for t in tools:
                table.add_row(t.name, t.description[:60])
            console.print(table)
        else:
            for t in tools:
                print(f"  {t.name}: {t.description}")

    elif command == "/recall":
        if not args:
            if use_rich:
                console.print("[yellow]Usage: /recall <query>[/yellow]")
        else:
            results = memory.search(args)
            if use_rich:
                from rich.markdown import Markdown
                console.print(Markdown(results))
            else:
                print(results)

    elif command == "/config":
        if use_rich:
            from rich.syntax import Syntax
            import yaml
            config_text = yaml.dump(config.model_dump(), default_flow_style=False, allow_unicode=True)
            console.print(Syntax(config_text, "yaml", theme="monokai"))
        else:
            print(config.model_dump())

    elif command == "/stats":
        stats = agent.get_stats()
        mem_stats = memory.get_stats()
        proc_stats = procedural.get_stats()
        if use_rich:
            console.print(f"Session: {stats}")
            console.print(f"Memory: {mem_stats}")
            console.print(f"Procedural: {proc_stats}")
        else:
            print(f"Session: {stats}")
            print(f"Memory: {mem_stats}")
            print(f"Procedural: {proc_stats}")

    # === New v0.2.0 commands ===

    elif command == "/health":
        healthy, issues = guardian.health.check_all()
        if use_rich:
            from rich.table import Table
            if healthy:
                console.print("[green]All systems healthy[/green]")
            else:
                console.print(f"[red]Found {len(issues)} issue(s):[/red]")
                for issue in issues:
                    console.print(f"  [yellow]- {issue}[/yellow]")

            # Individual check details
            table = Table(title="Health Checks")
            table.add_column("Check", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Issues", style="yellow")

            checks = [
                ("Config", guardian.health.check_config()),
                ("Memory", guardian.health.check_memory()),
                ("Tools", guardian.health.check_tools()),
                ("Soul", guardian.health.check_soul()),
                ("Providers", guardian.health.check_providers()),
            ]
            for name, issues_list in checks:
                status = "[green]OK[/green]" if not issues_list else "[red]FAIL[/red]"
                issue_text = "; ".join(issues_list) if issues_list else "-"
                table.add_row(name, status, issue_text)
            console.print(table)
        else:
            if healthy:
                print("All systems healthy")
            else:
                print(f"Found {len(issues)} issue(s):")
                for issue in issues:
                    print(f"  - {issue}")

    elif command == "/snapshot":
        tag = args.strip() if args else None
        snap_tag = guardian.snapshot_mgr.create_snapshot(tag)
        if use_rich:
            console.print(f"[green]Snapshot created: {snap_tag}[/green]")
        else:
            print(f"Snapshot created: {snap_tag}")

    elif command == "/snapshots":
        snapshots = guardian.snapshot_mgr.list_snapshots()
        if not snapshots:
            if use_rich:
                console.print("[yellow]No snapshots found[/yellow]")
            else:
                print("No snapshots found")
        elif use_rich:
            from rich.table import Table
            table = Table(title="Snapshots")
            table.add_column("Tag", style="cyan")
            table.add_column("Created", style="green")
            table.add_column("Size", style="yellow")
            for snap in snapshots:
                size_kb = snap["size_bytes"] / 1024
                table.add_row(
                    snap["tag"],
                    snap["created"][:19],
                    f"{size_kb:.1f} KB",
                )
            console.print(table)
        else:
            for snap in snapshots:
                size_kb = snap["size_bytes"] / 1024
                print(f"  {snap['tag']} | {snap['created'][:19]} | {size_kb:.1f} KB")

    elif command == "/rollback":
        tag = args.strip()
        if not tag:
            # Default to latest
            tag = guardian.snapshot_mgr.get_latest_snapshot_tag()
            if not tag:
                if use_rich:
                    console.print("[red]No snapshots available for rollback[/red]")
                else:
                    print("No snapshots available for rollback")
                return True

        if use_rich:
            console.print(f"[yellow]Rolling back to: {tag}...[/yellow]")
        else:
            print(f"Rolling back to: {tag}...")

        success = guardian.snapshot_mgr.restore_snapshot(tag)
        if success:
            if use_rich:
                console.print(f"[green]Successfully restored snapshot: {tag}[/green]")
                console.print("[yellow]Please restart Laimiu for changes to take full effect.[/yellow]")
            else:
                print(f"Successfully restored snapshot: {tag}")
                print("Please restart Laimiu for changes to take full effect.")
        else:
            if use_rich:
                console.print(f"[red]Failed to restore snapshot: {tag}[/red]")
            else:
                print(f"Failed to restore snapshot: {tag}")

    elif command == "/patterns":
        """Show pattern memory status."""
        patterns = procedural.tracker.get_all_patterns()
        if not patterns:
            if use_rich:
                console.print("[yellow]No patterns tracked yet[/yellow]")
            else:
                print("No patterns tracked yet")
        elif use_rich:
            from rich.table import Table
            table = Table(title="Tracked Patterns")
            table.add_column("Tool", style="cyan")
            table.add_column("Strength", style="green")
            table.add_column("Level", style="yellow")
            table.add_column("Occurrences", style="dim")
            table.add_column("Success Rate", style="dim")
            for p in patterns:
                table.add_row(
                    p.tool_name,
                    f"{p.strength:.2f}",
                    p.level,
                    str(p.occurrence),
                    f"{p.success_rate:.0%}",
                )
            console.print(table)
        else:
            for p in patterns:
                print(f"  {p.tool_name} | str={p.strength:.2f} | {p.level} | occ={p.occurrence} | {p.success_rate:.0%}")

    elif command == "/help":
        help_text = f"""
Available commands (v{__version__}):
  /model [name]       - Show or switch LLM provider
  /dream              - Run dream cycle (memory consolidation)
  /memory             - Show memory stats and index
  /tools              - List all registered tools
  /recall <query>     - Search memories
  /config             - Show current configuration
  /stats              - Show session statistics
  /health             - Show system health status
  /snapshot [tag]     - Create a state snapshot
  /snapshots          - List all snapshots
  /rollback [tag]     - Rollback to a snapshot
  /patterns           - Show tracked pattern memory
  /quit               - Exit Laimiu
"""
        if use_rich:
            console.print(help_text)
        else:
            print(help_text)

    else:
        if use_rich:
            console.print(f"[yellow]Unknown command: {command}. Type /help for available commands.[/yellow]")
        else:
            print(f"Unknown command: {command}. Type /help for available commands.")

    return True


def main() -> None:
    """Entry point for the Laimiu CLI."""
    # Setup logging
    from laimiu.utils.logging import setup_logging
    log_level = "DEBUG" if "--debug" in sys.argv else "INFO"
    setup_logging(log_level)

    # First-run setup wizard
    if _is_first_run():
        from laimiu.cli.setup import SetupWizard
        wizard = SetupWizard()
        config = wizard.run()
    else:
        # Initialize runtime (ensure dirs + defaults)
        _init_runtime()

        # Guardian: startup health check
        guardian = Guardian()
        guardian.startup_check()

        # Load config
        config = LaimiuConfig.load(CONFIG_FILE)

    # Run the async chat loop
    asyncio.run(_run_chat(config))


if __name__ == "__main__":
    main()
