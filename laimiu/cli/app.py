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
    return not CONFIG_FILE.exists()


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
                model="deepseek-v4-pro",
                api_key=os.environ["DEEPSEEK_API_KEY"],
            )
            config.provider.default = "deepseek"

        if os.environ.get("OPENAI_API_KEY"):
            from laimiu.config.settings import ProviderModelConfig
            models["glm"] = ProviderModelConfig(
                base_url="https://open.bigmodel.cn/api/coding/paas/v4",
                model="glm-4.7-flashx",
                api_key=os.environ["OPENAI_API_KEY"],
            )
            if not models.get("deepseek"):
                config.provider.default = "glm"

        if not models:
            # Default to Ollama
            from laimiu.config.settings import ProviderModelConfig
            models["ollama"] = ProviderModelConfig(
                base_url="http://localhost:11434/v1",
                model="llama3.1",
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
        from rich.rule import Rule
        from rich.table import Table
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

        banner = Table(show_header=False, box=None, padding=(0, 2))
        banner.add_row(
            Text("Laimiu", style="bold cyan"),
            Text(f"v{__version__}", style="dim"),
        )
        banner.add_row(
            Text("Model", style="dim"),
            Text(f"{provider_name}/{model_name}", style="green"),
        )
        banner.add_row(
            Text("Tools", style="dim"),
            Text(f"{builtin_count} registered", style="yellow"),
        )
        banner.add_row(
            Text("Memory", style="dim"),
            Text(f"{stats.get('vector_notes', 0)} notes", style="magenta"),
        )

        console.print()
        console.print(Rule(style="cyan"))
        console.print(Panel(banner, border_style="cyan", padding=(0, 1)))
        console.print(Rule(style="cyan"))
    except ImportError:
        print(f"Laimiu v{__version__}")
        print(f"Model: {config.provider.default}")
        print(f"Tools: {len(agent.tools.list_tools())}")


async def _run_chat(config: LaimiuConfig) -> None:
    """Main chat loop — dispatches to single or multi-agent mode."""
    if config.multi_agent.enabled:
        await _run_chat_multi(config)
    else:
        await _run_chat_single(config)


async def _run_chat_single(config: LaimiuConfig) -> None:
    """Original single-agent chat loop (unchanged behavior)."""
    try:
        from rich.console import Console
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

    discover_builtin_tools(registry)
    generated_count = discover_generated_tools(registry)

    from laimiu.tools.builtin.memory import MemoryRecallTool
    recall_tool = MemoryRecallTool(memory_manager=memory)
    registry.register(recall_tool)

    tracker = PatternTracker(config.procedural.extract_strength)
    procedural = ProceduralEngine(config, tracker, registry, router)
    reflection = Reflection()

    agent = AgentLoop(
        config=config, memory=memory, tool_registry=registry,
        router=router, reflection=reflection, procedural_tracker=tracker,
    )

    dream = DreamEngine(config, memory, procedural, router)
    guardian = Guardian()

    from laimiu.cli.renderer import ChatRenderer
    renderer = ChatRenderer(console) if use_rich else None

    if use_rich:
        _print_banner(config, agent, memory)
    else:
        print(f"Laimiu v{__version__} | Model: {config.provider.default}")

    if generated_count > 0:
        print(f"  Loaded {generated_count} learned tools")

    session_id = agent.start_session()

    provider_name = config.provider.default
    model_name = "?"
    if config.provider.models.get(provider_name):
        model_name = config.provider.models[provider_name].model
    tool_count = len(agent.tools.list_tools())

    def _status_bar():
        return f" Laimiu v{__version__} | {provider_name}/{model_name} | {tool_count} tools | /help "

    history_file = LAIMIU_HOME / "chat_history"
    if use_rich:
        session = PromptSession(
            history=FileHistory(str(history_file)),
            bottom_toolbar=_status_bar,
        )
    else:
        session = None

    while True:
        try:
            if use_rich:
                user_input = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: session.prompt("Laimiu> ", auto_suggest=AutoSuggestFromHistory())
                )
            else:
                user_input = input("Laimiu> ")

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                should_continue = await _handle_command(
                    user_input, config, agent, memory, dream, procedural,
                    guardian, console, use_rich
                )
                if not should_continue:
                    break
                continue

            if renderer:
                await renderer.render_turn(user_input, agent)
            else:
                print("Thinking...", flush=True)
                full_response = ""
                async for msg in agent.run(user_input):
                    if msg.type == "content":
                        print(msg.content, end="", flush=True)
                        full_response += msg.content
                    elif msg.type == "tool_call":
                        print(f"  [Calling {msg.content}...]", flush=True)
                    elif msg.type == "tool_result":
                        status = "OK" if msg.metadata.get("success") else "FAIL"
                        print(f"  [{msg.content} {status}]", flush=True)
                if full_response:
                    print()
                print("-" * 40)

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

    agent.end_session()
    dream.increment_sessions()

    if dream.should_dream():
        if use_rich:
            console.print("[dim]Running dream cycle...[/dim]")
        dream_results = await dream.dream()
        if use_rich and not dream_results.get("skipped"):
            console.print("[dim]Dream cycle completed[/dim]")


async def _run_chat_multi(config: LaimiuConfig) -> None:
    """Non-blocking multi-agent chat loop."""
    from laimiu.core.agents import AgentOrchestrator, AgentRole, AgentWorker
    from laimiu.core.message_bus import MessageBus
    from laimiu.cli.renderer import ChatRenderer

    try:
        from rich.console import Console
        from rich.panel import Panel as RichPanel
        from rich.text import Text as RichText
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

        console = Console()
        use_rich = True
    except ImportError:
        console = None
        use_rich = False
        RichPanel = None  # type: ignore[assignment,misc]
        RichText = None  # type: ignore[assignment,misc]

    # -- Shared infrastructure ----------------------------------------
    router = ProviderRouter(config)
    memory = MemoryManager(config)
    registry = ToolRegistry()

    discover_builtin_tools(registry)
    generated_count = discover_generated_tools(registry)

    from laimiu.tools.builtin.memory import MemoryRecallTool
    recall_tool = MemoryRecallTool(memory_manager=memory)
    registry.register(recall_tool)

    tracker = PatternTracker(config.procedural.extract_strength)
    procedural = ProceduralEngine(config, tracker, registry, router)
    reflection = Reflection()

    dream = DreamEngine(config, memory, procedural, router)
    guardian = Guardian()

    # -- Build agent workers from config ------------------------------
    message_bus = MessageBus()
    orchestrator = AgentOrchestrator(message_bus)
    renderer = ChatRenderer(console) if use_rich else None

    ma_cfg = config.multi_agent
    if not ma_cfg.agents:
        # Default: brain only
        brain_agent = AgentLoop(
            config=config, memory=memory, tool_registry=registry,
            router=router, reflection=reflection, procedural_tracker=tracker,
        )
        orchestrator.register_worker(AgentWorker(
            name="brain", role=AgentRole.BRAIN, agent=brain_agent,
        ))
    else:
        for name, role_cfg in ma_cfg.agents.items():
            if not role_cfg.enabled:
                continue
            agent = AgentLoop(
                config=config, memory=memory, tool_registry=registry,
                router=router, reflection=reflection, procedural_tracker=tracker,
            )
            # Set the router task so worker agents use the cheap provider
            agent._router_task = role_cfg.router_task
            role_enum = AgentRole.BRAIN if role_cfg.role == "brain" else AgentRole.WORKER
            orchestrator.register_worker(AgentWorker(
                name=name, role=role_enum, agent=agent,
            ))

    # -- Banner -------------------------------------------------------
    brain_worker = orchestrator.workers.get("brain")
    if use_rich and brain_worker:
        _print_banner(config, brain_worker.agent, memory)
        console.print(f"  [cyan]Multi-agent mode[/cyan]: {', '.join(orchestrator.workers.keys())}")
    else:
        print(f"Laimiu v{__version__} | Multi-agent mode")

    if generated_count > 0:
        print(f"  Loaded {generated_count} learned tools")

    orchestrator.start_sessions()

    # -- Status bar ---------------------------------------------------
    def _multi_status_bar():
        parts = []
        for w in orchestrator.workers.values():
            state = "busy" if w.is_busy else "idle"
            parts.append(f"{w.name}:{state}")
        return f" Laimiu v{__version__} | {' | '.join(parts)} | /help "

    history_file = LAIMIU_HOME / "chat_history"
    session: PromptSession | None = None
    if use_rich:
        session = PromptSession(
            history=FileHistory(str(history_file)),
            bottom_toolbar=_multi_status_bar,
        )

    # -- Start background worker loop ---------------------------------
    worker_loop_task = asyncio.create_task(orchestrator.run_worker_loop())
    output_queue = orchestrator.get_output_queue()

    # -- Non-blocking main loop ---------------------------------------
    loop = asyncio.get_running_loop()

    async def _read_input() -> str:
        if session:
            return await loop.run_in_executor(
                None,
                lambda: session.prompt("Laimiu> ", auto_suggest=AutoSuggestFromHistory())
            )
        return await loop.run_in_executor(None, lambda: input("Laimiu> "))

    input_future: asyncio.Future | None = asyncio.ensure_future(_read_input())
    output_future: asyncio.Future = asyncio.ensure_future(
        message_bus.get_next_output(output_queue, timeout=0.15)
    )

    # Per-task content accumulation
    _task_buffers: dict[str, str] = {}  # task_id -> accumulated content
    _task_tools: dict[str, list] = {}   # task_id -> tool log entries

    try:
        while True:
            wait_set = {f for f in [input_future, output_future] if f is not None}
            done, _pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

            # -- User input ready -------------------------------------
            if input_future and input_future in done:
                try:
                    user_input = input_future.result()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    input_future = asyncio.ensure_future(_read_input())
                    continue
                except Exception:
                    input_future = asyncio.ensure_future(_read_input())
                    continue

                # Immediately restart input reader
                input_future = asyncio.ensure_future(_read_input())

                user_input = user_input.strip()
                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    if user_input.lower() in ("/quit", "/exit", "/q"):
                        break
                    elif user_input.lower() == "/agents":
                        status = orchestrator.get_status()
                        if use_rich:
                            from rich.table import Table
                            table = Table(title="Agent Status")
                            table.add_column("Agent", style="cyan")
                            table.add_column("Role", style="green")
                            table.add_column("State", style="yellow")
                            table.add_column("Task", style="dim")
                            for name, info in status.items():
                                state = "[red]busy[/red]" if info["busy"] else "[green]idle[/green]"
                                task = info["current_task"] or "-"
                                table.add_row(name, info["role"], state, task)
                            console.print(table)
                        else:
                            for name, info in status.items():
                                print(f"  {name} ({info['role']}): {'busy' if info['busy'] else 'idle'}")
                        continue
                    elif user_input.lower().startswith("/bg "):
                        bg_msg = user_input[4:].strip()
                        if bg_msg:
                            task = orchestrator.route_to_worker(bg_msg)
                            if use_rich:
                                console.print(f"[yellow]Worker task {task.id}: {bg_msg[:50]}...[/yellow]")
                            else:
                                print(f"Worker task {task.id}: {bg_msg[:50]}...")
                        continue
                    elif user_input.lower() == "/help":
                        help_text = f"""
Available commands (v{__version__}, multi-agent):
  /model [name]       - Show or switch LLM provider
  /agents             - Show agent pool status
  /bg <message>       - Send task to worker agent (non-blocking)
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
                        continue
                    else:
                        # Delegate other commands to brain worker
                        brain_w = orchestrator.workers.get("brain")
                        if brain_w and brain_w.agent:
                            should_exit = await _handle_command(
                                user_input, config, brain_w.agent, memory,
                                dream, procedural, guardian, console, use_rich,
                            )
                            if not should_exit:
                                break
                        continue

                # Route regular message to brain (non-blocking)
                orchestrator.route_message(user_input)

                # Show user message panel
                if renderer and RichPanel and RichText:
                    console.print()
                    console.print(
                        RichPanel(
                            RichText(user_input),
                            title="[bold]You[/bold]",
                            border_style="blue",
                            padding=(0, 1),
                        )
                    )

            # -- Agent output ready ------------------------------------
            if output_future in done:
                try:
                    bus_msg = output_future.result()
                except Exception:
                    bus_msg = None

                if bus_msg and renderer:
                    tid = bus_msg.task_id
                    if bus_msg.type == "content":
                        # Accumulate content chunks per task
                        _task_buffers.setdefault(tid, "")
                        _task_buffers[tid] += bus_msg.content
                    elif bus_msg.type == "task_done":
                        # Task complete — render accumulated output
                        accumulated = _task_buffers.pop(tid, "")
                        _task_tools.pop(tid, None)
                        if accumulated:
                            from laimiu.core.message_bus import BusMessage as _BM
                            renderer.render_agent_output(_BM(
                                source=bus_msg.source,
                                task_id=tid,
                                type="content",
                                content=accumulated,
                            ))
                        renderer.render_agent_separator()
                    elif bus_msg.type == "thinking":
                        # Show thinking indicator
                        renderer.render_agent_output(bus_msg)
                    elif bus_msg.type in ("tool_call", "tool_result", "error"):
                        # Show tool/error output inline
                        renderer.render_agent_output(bus_msg)
                    elif bus_msg.type == "system":
                        renderer.render_agent_output(bus_msg)

                # Restart output reader
                output_future = asyncio.ensure_future(
                    message_bus.get_next_output(output_queue, timeout=0.15)
                )

    finally:
        orchestrator.stop()
        worker_loop_task.cancel()
        try:
            await worker_loop_task
        except asyncio.CancelledError:
            pass

    # -- Cleanup ------------------------------------------------------
    orchestrator.end_sessions()
    dream.increment_sessions()

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
  /agents             - Show agent pool status (multi-agent)
  /bg <message>       - Send task to worker agent (multi-agent)
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
