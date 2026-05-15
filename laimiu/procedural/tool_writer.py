"""Tool writer - generates Python scripts and registers them as tools."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from laimiu.constants import TOOLS_DIR
from laimiu.tools.registry import ToolRegistry

logger = logging.getLogger("laimiu.procedural.tool_writer")


class ToolWriter:
    """Generates Python tool scripts from LLM output and registers them."""

    def __init__(self, registry: ToolRegistry, tools_dir: Path | None = None):
        self.registry = registry
        self.tools_dir = tools_dir or TOOLS_DIR
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    async def write_and_register(
        self,
        script_content: str,
        tool_name: str | None = None,
    ) -> str | None:
        """Write a tool script to disk and register it.

        Args:
            script_content: The Python script content (may include markdown fencing).
            tool_name: Override tool name (otherwise extracted from TOOL_META).

        Returns:
            The tool name if successful, None if failed.
        """
        # Clean up markdown fencing
        script = self._clean_script(script_content)
        if not script:
            logger.error("Empty script content")
            return None

        # Extract tool name from TOOL_META if not provided
        if not tool_name:
            tool_name = self._extract_tool_name(script)
            if not tool_name:
                logger.error("Could not extract tool name from TOOL_META")
                return None

        # Validate before writing
        from laimiu.procedural.validator import ToolValidator

        validator = ToolValidator()
        is_valid, issues = validator.validate_script(script)
        if not is_valid:
            logger.error(f"Tool script validation failed: {issues}")
            return None

        # Write to file
        safe_name = re.sub(r"[^a-z0-9_]", "_", tool_name.lower())
        script_path = self.tools_dir / f"{safe_name}.py"

        script_path.write_text(script, encoding="utf-8")
        logger.info(f"Written tool script: {script_path}")

        # Register the tool directly (no need to re-scan filesystem)
        self._register_script(tool_name, script)

        return tool_name

    def _register_script(self, tool_name: str, script: str) -> None:
        """Parse and register a tool script directly."""
        import importlib.util
        import sys

        safe_name = re.sub(r"[^a-z0-9_]", "_", tool_name.lower())
        module_name = f"_laimiu_tool_{safe_name}"
        script_path = self.tools_dir / f"{safe_name}.py"

        try:
            spec = importlib.util.spec_from_file_location(module_name, str(script_path))
            if spec is None or spec.loader is None:
                logger.error(f"Cannot create module spec for {script_path}")
                return
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Check for TOOL_META + run()
            if not hasattr(module, "TOOL_META") or not hasattr(module, "run"):
                logger.error(f"Tool script missing TOOL_META or run(): {tool_name}")
                return

            meta = module.TOOL_META
            run_fn = module.run

            # Create a dynamic BaseTool subclass
            from laimiu.tools.base import BaseTool, ToolResult
            from typing import Any as _Any

            class GeneratedTool(BaseTool):
                pass

            GeneratedTool.name = meta.get("name", tool_name)
            GeneratedTool.description = meta.get("description", "")
            GeneratedTool.parameters = meta.get("parameters", {"type": "object", "properties": {}})

            async def _execute(self, **kwargs: _Any) -> ToolResult:
                try:
                    result = run_fn(**kwargs)
                    return ToolResult(success=True, output=str(result))
                except Exception as e:
                    return ToolResult(success=False, error=str(e))

            GeneratedTool.execute = _execute
            self.registry.register(GeneratedTool())
            logger.info(f"Registered generated tool: {tool_name}")

        except Exception as e:
            logger.error(f"Failed to register tool {tool_name}: {e}")

    def _clean_script(self, content: str) -> str:
        """Remove markdown code fencing if present."""
        # Remove ```python ... ``` fencing
        match = re.search(r"```(?:python)?\s*\n(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content.strip()

    def _extract_tool_name(self, script: str) -> str | None:
        """Extract tool name from TOOL_META in the script."""
        match = re.search(r'"name"\s*:\s*"([^"]+)"', script)
        if match:
            return match.group(1)
        return None

    def list_generated_tools(self) -> list[Path]:
        """List all generated tool scripts."""
        return sorted(self.tools_dir.glob("*.py"))
