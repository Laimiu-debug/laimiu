"""Python code execution tool — full Python environment with minimal safety."""

from __future__ import annotations

import sys
import traceback
from io import StringIO
from pathlib import Path
from typing import Any

from laimiu.tools.base import BaseTool, ToolResult


class CodeExecTool(BaseTool):
    name = "code_exec"
    description = "Execute Python code and return the output. Full stdlib access."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 10)",
                "default": 10,
            },
        },
        "required": ["code"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        code = kwargs.get("code", "")
        if not code:
            return ToolResult(success=False, error="No code provided")

        # Minimal safety: only block truly destructive operations
        for pattern in ["os.system(", "subprocess.call("]:
            if pattern in code:
                return ToolResult(success=False, error=f"Blocked: {pattern.strip('(')}")

        # Full Python environment — use real builtins
        exec_globals: dict[str, Any] = {"__builtins__": __builtins__}
        exec_locals: dict[str, Any] = {}

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture = StringIO()
        sys.stderr = stderr_capture = StringIO()

        try:
            exec(compile(code, "<code_exec>", "exec"), exec_globals, exec_locals)
            output = stdout_capture.getvalue()
            errors = stderr_capture.getvalue()

            if errors:
                output += f"\n[stderr] {errors}"

            return ToolResult(
                success=True,
                output=output if output else "(no output)",
                data={"locals": {k: repr(v) for k, v in exec_locals.items() if not k.startswith("_")}},
            )
        except Exception:
            tb = traceback.format_exc()
            return ToolResult(
                success=False,
                error=f"Execution error:\n{tb}",
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
