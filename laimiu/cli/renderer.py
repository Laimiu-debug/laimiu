"""Rich-based structured chat renderer for Laimiu.

Converts OutputMessage stream into a polished terminal UI with:
- User messages in blue panels
- Streaming Markdown for assistant replies (with syntax highlighting)
- Compact tool call status lines (name, elapsed, success/fail)
- Dim thinking indicator for reasoning models
- Separator lines between turns
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console

    from laimiu.core.agent import AgentLoop


class ChatRenderer:
    """Renders structured agent output with Rich formatting."""

    def __init__(self, console: Console):
        self.console = console

    async def render_turn(self, user_input: str, agent: AgentLoop) -> str:
        """Render a complete user -> agent turn. Returns assistant response text."""
        # -- User message panel --
        self.console.print()
        self.console.print(
            Panel(
                Text(user_input),
                title="[bold]You[/bold]",
                border_style="blue",
                padding=(0, 1),
            )
        )

        # -- Agent streaming response --
        full_response = ""
        tool_log: list[tuple[str, float, bool]] = []
        is_thinking = False

        with Live(
            console=self.console,
            refresh_per_second=12,
            vertical_overflow="visible",
            transient=False,
        ) as live:
            live.update(Text("Thinking...", style="dim italic"))

            async for msg in agent.run(user_input):
                if msg.type == "thinking":
                    is_thinking = True
                    live.update(Text("Thinking...", style="dim italic"))

                elif msg.type == "content":
                    full_response += msg.content
                    live.update(
                        self._build(full_response, tool_log, active_tool=None)
                    )
                    is_thinking = False

                elif msg.type == "tool_call":
                    live.update(
                        self._build(full_response, tool_log, active_tool=msg.content)
                    )

                elif msg.type == "tool_result":
                    elapsed = msg.metadata.get("elapsed_ms", 0)
                    success = msg.metadata.get("success", False)
                    tool_log.append((msg.content, elapsed, success))
                    live.update(
                        self._build(full_response, tool_log, active_tool=None)
                    )

                elif msg.type == "error":
                    self.console.print(
                        f"[red]Error: {msg.content}[/red]"
                    )

            # Final render
            live.update(self._build(full_response, tool_log, None))

        # Turn separator
        self.console.print(Rule(style="dim"))
        return full_response

    def _build(
        self,
        response: str,
        tool_log: list[tuple[str, float, bool]],
        active_tool: str | None,
    ) -> Group:
        """Build the composite Rich renderable for the current state."""
        parts: list = []

        # Tool call log
        for name, elapsed, success in tool_log:
            mark = "+" if success else "x"
            style = "green" if success else "red"
            parts.append(
                Text(f"  [{mark}] {name} ({elapsed:.0f}ms)", style=style)
            )

        # Active tool spinner
        if active_tool:
            parts.append(Text(f"  > {active_tool}...", style="yellow"))

        # Blank line between tools and content
        if (tool_log or active_tool) and response:
            parts.append(Text(""))

        # Main content
        if response:
            parts.append(Markdown(response))
        else:
            parts.append(Text("Thinking...", style="dim italic"))

        return Group(*parts)
