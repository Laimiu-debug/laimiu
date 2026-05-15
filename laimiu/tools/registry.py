"""Tool registry with AST discovery and auto-registration."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any

from laimiu.tools.base import BaseTool, ToolDefinition, ToolResult
from laimiu.constants import TOOLS_DIR

logger = logging.getLogger("laimiu.tools.registry")


class ToolRegistry:
    """Central registry for all tools (builtin + generated)."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        self._tools[tool.name] = tool
        self._definitions[tool.name] = tool.get_definition()
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)
        self._definitions.pop(name, None)

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    async def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given arguments."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        try:
            result = await tool.execute(**args)
            return result
        except Exception as e:
            logger.error(f"Tool '{name}' execution failed: {e}")
            return ToolResult(success=False, error=str(e))

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tool definitions."""
        return list(self._definitions.values())

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI function-calling format."""
        return [defn.to_openai_spec() for defn in self._definitions.values()]

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    @property
    def count(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())


def discover_builtin_tools(registry: ToolRegistry) -> None:
    """Discover and register all builtin tools via AST import."""
    builtin_dir = Path(__file__).parent / "builtin"
    _discover_tools_in_dir(registry, builtin_dir)


def discover_generated_tools(registry: ToolRegistry) -> int:
    """Discover and register agent-generated tools from ~/.laimiu/tools/.

    Returns the number of tools discovered.
    """
    if not TOOLS_DIR.exists():
        return 0
    count = registry.count
    _discover_tools_in_dir(registry, TOOLS_DIR)
    return registry.count - count


def _discover_tools_in_dir(registry: ToolRegistry, directory: Path) -> None:
    """Import all Python files in a directory and register any BaseTool subclasses."""
    if not directory.exists():
        return

    for py_file in sorted(directory.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = f"laimiu_tools_{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find BaseTool subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    inspect.isclass(attr)
                    and issubclass(attr, BaseTool)
                    and attr is not BaseTool
                    and attr.name
                ):
                    try:
                        registry.register(attr())
                    except Exception as e:
                        logger.error(f"Failed to register tool from {py_file}: {e}")

            # Check for TOOL_META-based tools (generated tools)
            if hasattr(module, "TOOL_META"):
                _register_meta_tool(registry, module)

        except Exception as e:
            logger.error(f"Failed to load tool file {py_file}: {e}")


def _register_meta_tool(registry: ToolRegistry, module: Any) -> None:
    """Register a tool defined with TOOL_META (generated tools)."""
    meta = module.TOOL_META
    if not isinstance(meta, dict) or "name" not in meta:
        return

    name = meta["name"]
    if registry.has_tool(name):
        return

    # Create a dynamic tool wrapper
    run_fn = getattr(module, "run", None)
    if run_fn is None:
        logger.warning(f"Generated tool '{name}' has no run() function")
        return

    class GeneratedTool(BaseTool):
        pass

    GeneratedTool.name = name
    GeneratedTool.description = meta.get("description", "")
    GeneratedTool.parameters = meta.get("parameters", {"type": "object", "properties": {}})

    async def _execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = run_fn(**kwargs)
            return ToolResult(success=True, output=str(result))
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    GeneratedTool.execute = _execute
    registry.register(GeneratedTool())
