"""Enhanced Rich-based structured chat renderer for Laimiu v2.

Design inspired by Claude Code + Hermes:
- Color-coded user/assistant/tool/thinking zones
- Per-response header with model, timing, tools used
- Streaming Markdown with syntax highlighting
- Animated spinner for active tool calls
- Scrollable thinking display (last N lines, not truncated chars)
- Clean separators between turns
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console

    from laimiu.core.agent import AgentLoop
    from laimiu.core.message_bus import BusMessage

# ── Color Palette ──────────────────────────────────────────────
C_USER_BORDER = "bright_blue"
C_USER_TITLE = "bold bright_blue"

C_AI_HEADER = "dim cyan"
C_AI_BORDER = "cyan"
C_AI_TITLE = "bold cyan"

C_TOOL_ACTIVE = "yellow"
C_TOOL_OK = "green"
C_TOOL_FAIL = "red"
C_TOOL_DIM = "dim"

C_THINKING = "dim italic"
C_ERROR = "bold red"
C_SEPARATOR = "dim"
C_SYSTEM = "dim magenta"

# Agent source → (panel border, title color) for multi-agent
AGENT_STYLES: dict[str, tuple[str, str]] = {
    "brain": ("cyan", "bold cyan"),
    "worker": ("yellow", "bold yellow"),
    "system": ("magenta", "bold magenta"),
}

# ── Icons ──────────────────────────────────────────────────────
ICON_TOOL = "\u2699"      # ⚙
ICON_OK = "\u2713"        # ✓
ICON_FAIL = "\u2717"      # ✗
ICON_THINK = "\u2026"     # …
ICON_CLOCK = "\u23f1"     # ⏱
ICON_BOX = "\U0001f4e6"   # 📦
ICON_BRAIN = "\U0001f9e0" # 🧠
ICON_LIGHTNING = "\u26a1"  # ⚡

# Thinking display limits
THINKING_MAX_LINES = 5


class ChatRenderer:
    """Renders structured agent output with enhanced Rich formatting."""

    def __init__(self, console: Console):
        self.console = console
        self.turn_count = 0
        self.session_start = time.perf_counter()

    # ── Single-agent turn render ──────────────────────────────

    async def render_turn(
        self,
        user_input: str,
        agent: AgentLoop,
        model_name: str = "",
        provider_name: str = "",
    ) -> str:
        """Render a complete user → agent turn. Returns full assistant response."""
        self.turn_count += 1

        # ── User message ──────────────────────────────────────
        self.console.print()
        self.console.print(
            Panel(
                Text(user_input, style="white"),
                title=f"[{C_USER_TITLE}]You[/{C_USER_TITLE}]",
                border_style=C_USER_BORDER,
                padding=(0, 1),
            )
        )

        # ── Agent streaming response ──────────────────────────
        full_response = ""
        tool_log: list[tuple[str, float, bool]] = []  # (name, elapsed_ms, success)
        is_thinking = False
        thinking_content = ""
        t_start = time.perf_counter()
        t_first_token: float | None = None

        with Live(
            console=self.console,
            refresh_per_second=15,
            vertical_overflow="visible",
            transient=False,
        ) as live:
            # Initial thinking indicator
            live.update(self._build_thinking())

            async for msg in agent.run(user_input):
                if msg.type == "thinking":
                    is_thinking = True
                    thinking_content = msg.content
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    live.update(self._build_thinking(msg.content))

                elif msg.type == "content":
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    full_response += msg.content
                    live.update(
                        self._build_response(
                            full_response, tool_log, active_tool=None,
                            thinking=is_thinking,
                        )
                    )
                    is_thinking = False

                elif msg.type == "tool_call":
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    live.update(
                        self._build_response(
                            full_response, tool_log, active_tool=msg.content,
                        )
                    )

                elif msg.type == "tool_result":
                    elapsed = msg.metadata.get("elapsed_ms", 0)
                    success = msg.metadata.get("success", False)
                    tool_log.append((msg.content, elapsed, success))
                    live.update(
                        self._build_response(
                            full_response, tool_log, active_tool=None,
                        )
                    )

                elif msg.type == "error":
                    self.console.print(
                        Panel(
                            Text(msg.content, style=C_ERROR),
                            title="[bold red]Error[/bold red]",
                            border_style="red",
                            padding=(0, 1),
                        )
                    )

            # Final render
            live.update(
                self._build_response(full_response, tool_log, active_tool=None)
            )

        # ── Timing ────────────────────────────────────────────
        t_end = time.perf_counter()
        total_time = t_end - t_start
        think_time = (t_first_token - t_start) if t_first_token else total_time

        # ── Footer: model + timing ────────────────────────────
        self._render_footer(
            model=model_name,
            provider=provider_name,
            think_time=think_time,
            total_time=total_time,
            tool_count=len(tool_log),
        )

        # ── Separator ─────────────────────────────────────────
        self.console.print(Rule(style=C_SEPARATOR))
        return full_response

    # ── Multi-agent output ────────────────────────────────────

    def render_agent_output(self, msg: BusMessage) -> None:
        """Render a BusMessage from a named agent source."""
        border, title_style = AGENT_STYLES.get(msg.source, ("white", "bold white"))
        source_label = f"[{title_style}]{msg.source}[/{title_style}]"

        if msg.type == "content":
            self.console.print()
            self.console.print(
                Panel(
                    Markdown(msg.content),
                    title=source_label,
                    border_style=border,
                    padding=(0, 1),
                )
            )
        elif msg.type == "thinking":
            self.console.print(
                Text(f"  [{msg.source}] {ICON_THINK} Thinking...", style=C_THINKING)
            )
        elif msg.type == "tool_call":
            self.console.print(
                Text(f"  {ICON_TOOL} [{msg.source}] \u2192 {msg.content}...", style=C_TOOL_ACTIVE)
            )
        elif msg.type == "tool_result":
            elapsed = msg.metadata.get("elapsed_ms", 0)
            success = msg.metadata.get("success", False)
            mark = ICON_OK if success else ICON_FAIL
            style = C_TOOL_OK if success else C_TOOL_FAIL
            self.console.print(
                Text(
                    f"  [{msg.source}] [{mark}] {msg.content} ({elapsed:.0f}ms)",
                    style=style,
                )
            )
        elif msg.type == "error":
            self.console.print(
                Panel(
                    Text(msg.content, style=C_ERROR),
                    title=f"[bold red]{msg.source}[/bold red]",
                    border_style="red",
                    padding=(0, 1),
                )
            )
        elif msg.type == "system":
            self.console.print(
                Text(f"  [{msg.source}] {msg.content}", style=C_SYSTEM)
            )

    def render_agent_separator(self) -> None:
        """Print a separator after an agent's complete output."""
        self.console.print(Rule(style=C_SEPARATOR))

    # ── Internal builders ─────────────────────────────────────

    def _build_thinking(self, content: str = "") -> Group:
        """Build thinking indicator with scrollable reasoning text.

        Shows the last N lines of reasoning (not truncated by character count).
        """
        parts: list = []
        if content:
            lines = content.splitlines()
            # Keep last N lines to avoid flooding the terminal
            if len(lines) > THINKING_MAX_LINES:
                display_lines = lines[-THINKING_MAX_LINES:]
                snippet = "... (earlier reasoning hidden)\n" + "\n".join(display_lines)
            else:
                snippet = content

            token_hint = f"Thinking \u00b7 {len(content)} chars"
            parts.append(
                Panel(
                    Text(snippet, style=C_THINKING),
                    title=f"[{C_THINKING}]{token_hint}[/{C_THINKING}]",
                    border_style="dim",
                    padding=(0, 1),
                )
            )
        else:
            parts.append(
                Spinner("dots", Text(f"  {ICON_THINK} Thinking...", style=C_THINKING))
            )
        return Group(*parts)

    def _build_response(
        self,
        content: str,
        tool_log: list[tuple[str, float, bool]],
        active_tool: str | None = None,
        thinking: bool = False,
    ) -> Group:
        """Build composite renderable: tool log + content."""
        parts: list = []

        # ── Tool log (completed) ─────────────────────────────
        for name, elapsed, success in tool_log:
            mark = ICON_OK if success else ICON_FAIL
            style = C_TOOL_OK if success else C_TOOL_FAIL
            parts.append(
                Text(f"  {ICON_TOOL} {name}  {mark} {elapsed:.0f}ms", style=style)
            )

        # ── Active tool (animated spinner) ───────────────────
        if active_tool:
            parts.append(
                Spinner("dots", Text(f"  {ICON_TOOL} {active_tool}", style=C_TOOL_ACTIVE))
            )

        # ── Spacing ──────────────────────────────────────────
        if (tool_log or active_tool) and content:
            parts.append(Text(""))

        # ── Main content ─────────────────────────────────────
        if content:
            parts.append(Markdown(content))
        else:
            parts.append(
                Spinner("dots", Text(f"  {ICON_THINK} Thinking...", style=C_THINKING))
            )

        return Group(*parts)

    def _render_footer(
        self,
        model: str,
        provider: str = "",
        think_time: float = 0.0,
        total_time: float = 0.0,
        tool_count: int = 0,
    ) -> None:
        """Render the timing/model footer after each response."""
        parts: list[Text] = []

        if provider and model:
            parts.append(Text.from_markup(
                f"[{C_AI_HEADER}]{provider}/{model}[/{C_AI_HEADER}]"
            ))
        elif model:
            parts.append(Text.from_markup(
                f"[{C_AI_HEADER}]{model}[/{C_AI_HEADER}]"
            ))

        parts.append(Text.from_markup(
            f"[{C_TOOL_DIM}]{ICON_CLOCK} think {think_time:.1f}s[/{C_TOOL_DIM}]"
        ))
        parts.append(Text.from_markup(
            f"[{C_TOOL_DIM}]{ICON_CLOCK} total {total_time:.1f}s[/{C_TOOL_DIM}]"
        ))

        if tool_count > 0:
            parts.append(Text.from_markup(
                f"[{C_TOOL_DIM}]{ICON_BOX} {tool_count} tools[/{C_TOOL_DIM}]"
            ))

        separator = Text(" \u00b7 ", style=C_TOOL_DIM)
        footer = separator.join(parts)
        self.console.print(footer)
        self.console.print()  # spacing before separator
