"""File read/write/search tools."""

from __future__ import annotations

import glob as glob_mod
from pathlib import Path
from typing import Any

from laimiu.tools.base import BaseTool, ToolResult
from laimiu.utils.safety import sanitize_path, is_source_write_protected


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the contents of a file"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (0-based)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read",
            },
        },
        "required": ["path"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", "")
        if not path:
            return ToolResult(success=False, error="No path provided")
        if not sanitize_path(path):
            return ToolResult(success=False, error="Path not allowed")

        file_path = Path(path)
        if not file_path.exists():
            return ToolResult(success=False, error=f"File not found: {path}")
        if not file_path.is_file():
            return ToolResult(success=False, error=f"Not a file: {path}")

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            offset = kwargs.get("offset", 0) or 0
            limit = kwargs.get("limit")

            selected = lines[offset:]
            if limit:
                selected = selected[:limit]

            result = "".join(
                f"{i + offset + 1}: {line}" for i, line in enumerate(selected)
            )
            total = len(lines)
            shown = len(selected)
            header = f"File: {path} ({total} lines"
            if offset or limit:
                header += f", showing {shown}"
            header += ")\n"

            return ToolResult(success=True, output=header + result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file (creates parent dirs if needed)"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write",
            },
            "append": {
                "type": "boolean",
                "description": "If true, append instead of overwrite",
                "default": False,
            },
        },
        "required": ["path", "content"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        append = kwargs.get("append", False)

        if not path:
            return ToolResult(success=False, error="No path provided")
        if not sanitize_path(path):
            return ToolResult(success=False, error="Path not allowed")
        if is_source_write_protected(path):
            return ToolResult(
                success=False,
                error="Cannot write to Laimiu's own *.py source files or pyproject.toml. "
                      "You can write to any other file: config, data, docs, scripts, etc.",
            )

        file_path = Path(path)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)
            action = "Appended to" if append else "Wrote"
            return ToolResult(
                success=True,
                output=f"{action} {path} ({len(content)} chars)",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SearchFilesTool(BaseTool):
    name = "search_files"
    description = "Search for files matching a glob pattern"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match (e.g. '**/*.py')",
            },
            "directory": {
                "type": "string",
                "description": "Directory to search in (default: current)",
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        pattern = kwargs.get("pattern", "")
        directory = kwargs.get("directory", ".")

        if not pattern:
            return ToolResult(success=False, error="No pattern provided")

        try:
            matches = list(Path(directory).glob(pattern))
            if not matches:
                return ToolResult(success=True, output="No files matched")
            lines = [str(m) for m in sorted(matches)[:100]]
            output = "\n".join(lines)
            if len(matches) > 100:
                output += f"\n... and {len(matches) - 100} more"
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GrepFilesTool(BaseTool):
    name = "grep_files"
    description = "Search file contents for a pattern (like grep)"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "directory": {
                "type": "string",
                "description": "Directory to search in",
            },
            "file_pattern": {
                "type": "string",
                "description": "Glob pattern for files to search (e.g. '*.py')",
                "default": "**/*",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results",
                "default": 50,
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        import re

        pattern = kwargs.get("pattern", "")
        directory = kwargs.get("directory", ".")
        file_pattern = kwargs.get("file_pattern", "**/*")
        max_results = kwargs.get("max_results", 50)

        if not pattern:
            return ToolResult(success=False, error="No pattern provided")

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult(success=False, error=f"Invalid regex: {e}")

        results = []
        try:
            for fpath in Path(directory).glob(file_pattern):
                if not fpath.is_file():
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append(f"{fpath}:{i}: {line.rstrip()}")
                                if len(results) >= max_results:
                                    break
                except (PermissionError, OSError):
                    continue
                if len(results) >= max_results:
                    break
        except Exception as e:
            return ToolResult(success=False, error=str(e))

        if not results:
            return ToolResult(success=True, output="No matches found")

        output = "\n".join(results)
        return ToolResult(success=True, output=output)
