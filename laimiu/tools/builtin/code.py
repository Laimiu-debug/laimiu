"""Python code execution sandbox tool."""

from __future__ import annotations

import builtins
import sys
import traceback
from io import StringIO
from pathlib import Path
from typing import Any

from laimiu.tools.base import BaseTool, ToolResult


def _make_safe_open() -> Any:
    """Create an open() wrapper that blocks writes to Laimiu source files."""
    _real_open = builtins.open

    # Resolve source dir once
    import laimiu as _pkg
    _src_dir = str(Path(_pkg.__file__).parent.resolve())
    _project_root = str(Path(_pkg.__file__).parent.parent.resolve())

    def safe_open(file, mode="r", *args, **kwargs):
        # Block write modes to source files
        if isinstance(file, (str, Path)):
            resolved = str(Path(file).resolve())
            # Protect laimiu/*.py source and pyproject.toml
            if resolved.startswith(_src_dir) and resolved.endswith(".py") and ("w" in mode or "a" in mode):
                raise PermissionError(
                    f"Cannot write to Laimiu source: {file}. "
                    f"You can write to any other file."
                )
            if resolved == str(Path(_project_root) / "pyproject.toml") and ("w" in mode or "a" in mode):
                raise PermissionError("Cannot write to pyproject.toml from sandbox.")
        return _real_open(file, mode, *args, **kwargs)

    return safe_open


class CodeExecTool(BaseTool):
    name = "code_exec"
    description = "Execute Python code in a sandbox and return the output. Supports imports."
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

        # Safety check: block explicitly dangerous patterns
        from laimiu.utils.safety import is_code_safe

        safe, reason = is_code_safe(code)
        if not safe:
            return ToolResult(success=False, error=f"Code blocked: {reason}")

        # Build sandbox with __import__ for module access
        restricted_globals = {
            "__builtins__": {
                # Core I/O
                "print": print,
                "open": _make_safe_open(),  # Protected: can't write source files
                # Types
                "str": str, "int": int, "float": float, "bool": bool,
                "list": list, "dict": dict, "tuple": tuple, "set": set,
                "bytes": bytes, "bytearray": bytearray,
                "type": type, "repr": repr, "format": format,
                "isinstance": isinstance, "issubclass": issubclass,
                "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
                # Import — the key unlock
                "__import__": __import__,
                # Math
                "abs": abs, "min": min, "max": max, "sum": sum,
                "round": round, "pow": pow, "sorted": sorted,
                # Iteration
                "range": range, "enumerate": enumerate, "zip": zip,
                "map": map, "filter": filter, "reversed": reversed,
                "iter": iter, "next": next, "slice": slice,
                "any": any, "all": all,
                # Conversion
                "ord": ord, "chr": chr, "hex": hex, "oct": oct, "bin": bin,
                "id": id, "hash": hash, "len": len,
                # Object
                "object": object, "super": super, "property": property,
                "staticmethod": staticmethod, "classmethod": classmethod,
                # Exceptions
                "Exception": Exception, "ValueError": ValueError,
                "TypeError": TypeError, "KeyError": KeyError,
                "IndexError": IndexError, "RuntimeError": RuntimeError,
                "ImportError": ImportError, "ModuleNotFoundError": ModuleNotFoundError,
                "AttributeError": AttributeError, "NameError": NameError,
                "FileNotFoundError": FileNotFoundError, "PermissionError": PermissionError,
                "OSError": OSError, "TimeoutError": TimeoutError,
                "StopIteration": StopIteration, "NotImplementedError": NotImplementedError,
                # Constants
                "True": True, "False": False, "None": None, "Ellipsis": Ellipsis,
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
