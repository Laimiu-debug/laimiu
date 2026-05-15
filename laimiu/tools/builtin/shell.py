"""Shell execution tool."""

from __future__ import annotations

import asyncio
from typing import Any

from laimiu.tools.base import BaseTool, ToolResult
from laimiu.utils.safety import is_command_dangerous


class ShellTool(BaseTool):
    name = "shell"
    description = "Execute a shell command and return its output"
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 30)",
                "default": 30,
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the command",
            },
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs.get("command", "")
        timeout = kwargs.get("timeout", 30)
        cwd = kwargs.get("cwd")

        if not command:
            return ToolResult(success=False, error="No command provided")

        if is_command_dangerous(command):
            return ToolResult(
                success=False,
                error=f"Command blocked for safety: contains dangerous pattern. "
                f"Run manually if you're sure.",
            )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            output_parts = []
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                output_parts.append(f"[stderr] {stderr.decode('utf-8', errors='replace')}")

            output = "\n".join(output_parts) if output_parts else "(no output)"

            if proc.returncode != 0:
                return ToolResult(
                    success=False,
                    output=output,
                    error=f"Exit code: {proc.returncode}",
                )
            return ToolResult(success=True, output=output)

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout}s",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
