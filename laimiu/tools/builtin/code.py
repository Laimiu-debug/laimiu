"""Python code execution sandbox tool."""

from __future__ import annotations

import sys
import traceback
from io import StringIO
from typing import Any

from laimiu.tools.base import BaseTool, ToolResult


class CodeExecTool(BaseTool):
    name = "code_exec"
    description = "Execute Python code in a restricted sandbox and return the output"
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

        # Safety check: block dangerous operations
        from laimiu.utils.safety import is_code_safe

        safe, reason = is_code_safe(code)
        if not safe:
            return ToolResult(success=False, error=f"Code blocked: {reason}")

        # Execute in restricted namespace
        restricted_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "bool": bool,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "isinstance": isinstance,
                "type": type,
                "hasattr": hasattr,
                "getattr": getattr,
                "repr": repr,
                "round": round,
                "any": any,
                "all": all,
                "open": open,  # Needed for file ops, safety checked above
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "KeyError": KeyError,
                "IndexError": IndexError,
                "RuntimeError": RuntimeError,
                "ImportError": ImportError,
            },
        }
        restricted_locals: dict[str, Any] = {}

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture = StringIO()
        sys.stderr = stderr_capture = StringIO()

        try:
            exec(compile(code, "<code_exec>", "exec"), restricted_globals, restricted_locals)
            output = stdout_capture.getvalue()
            errors = stderr_capture.getvalue()

            if errors:
                output += f"\n[stderr] {errors}"

            return ToolResult(
                success=True,
                output=output if output else "(no output)",
                data={"locals": {k: repr(v) for k, v in restricted_locals.items() if not k.startswith("_")}},
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
