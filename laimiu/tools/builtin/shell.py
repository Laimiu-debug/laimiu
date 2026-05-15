"""Shell execution tool — works on Windows and Linux."""

from __future__ import annotations

import asyncio
import sys
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
                error=f"Command blocked for safety: contains dangerous pattern.",
            )

        try:
            # Windows needs shell=True and proper encoding
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                # On Windows, use cmd.exe explicitly
                **({"shell": True} if sys.platform == "win32" else {}),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            # Decode with fallback for Windows
            encoding = "utf-8"
            output_parts = []
            if stdout:
                output_parts.append(stdout.decode(encoding, errors="replace"))
            if stderr:
                stderr_text = stderr.decode(encoding, errors="replace")
                # Don't treat stderr as failure if exit code is 0
                if proc.returncode == 0:
                    output_parts.append(f"[stderr] {stderr_text}")
                else:
                    output_parts.append(f"[stderr] {stderr_text}")

            output = "\n".join(output_parts) if output_parts else "(no output)"

            if proc.returncode != 0:
                return ToolResult(
                    success=False,
                    output=output,
                    error=f"Exit code: {proc.returncode}",
                )
            return ToolResult(success=True, output=output)

        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout}s",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
