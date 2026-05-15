"""Shell execution tool — works on Windows and Linux.

Uses subprocess.Popen wrapped in run_in_executor for cross-platform reliability.
Avoids asyncio.create_subprocess_shell issues on Windows.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import Any

from laimiu.tools.base import BaseTool, ToolResult
from laimiu.utils.safety import is_command_dangerous


def _run_command(command: str, timeout: int, cwd: str | None) -> tuple[int, str, str]:
    """Run a shell command synchronously and return (returncode, stdout, stderr)."""
    import subprocess as _sp
    proc = _sp.Popen(
        command,
        shell=True,
        stdout=_sp.PIPE,
        stderr=_sp.PIPE,
        cwd=cwd,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except _sp.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return -1, stdout.decode("utf-8", errors="replace"), "Command timed out"

    return (
        proc.returncode,
        stdout.decode("utf-8", errors="replace") if stdout else "",
        stderr.decode("utf-8", errors="replace") if stderr else "",
    )


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
                error="Command blocked for safety: contains dangerous pattern.",
            )

        try:
            loop = asyncio.get_running_loop()
            returncode, stdout, stderr = await loop.run_in_executor(
                None, _run_command, command, timeout, cwd
            )

            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                if returncode == 0:
                    output_parts.append(f"[stderr] {stderr}")
                else:
                    output_parts.append(f"[stderr] {stderr}")

            output = "\n".join(output_parts) if output_parts else "(no output)"

            if returncode != 0:
                if "timed out" in stderr:
                    return ToolResult(
                        success=False,
                        output=output,
                        error=f"Command timed out after {timeout}s",
                    )
                return ToolResult(
                    success=False,
                    output=output,
                    error=f"Exit code: {returncode}",
                )
            return ToolResult(success=True, output=output)

        except Exception as e:
            return ToolResult(success=False, error=str(e))
