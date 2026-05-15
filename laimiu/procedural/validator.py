"""Tool validator - ensures generated tools are safe and well-formed."""

from __future__ import annotations

import ast
import logging
from typing import Any

from laimiu.utils.safety import is_code_safe

logger = logging.getLogger("laimiu.procedural.validator")


class ToolValidator:
    """Validates that agent-generated tool scripts are safe and usable.

    Checks:
    1. Static analysis: no forbidden patterns (os.system, eval, exec, etc.)
    2. Structure check: must have TOOL_META and run() function
    3. Syntax check: must be valid Python
    """

    FORBIDDEN_PATTERNS = [
        "os.system",
        "subprocess.call",
        "subprocess.Popen",
        "__import__",
        "eval(",
        "exec(",
        "compile(",
        "os.remove",
        "shutil.rmtree",
    ]

    def validate_script(self, source: str) -> tuple[bool, list[str]]:
        """Validate a tool script.

        Returns (is_valid, list_of_issues).
        """
        issues = []

        # 1. Syntax check
        try:
            ast.parse(source)
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
            return False, issues

        # 2. Safety check
        safe, reason = is_code_safe(source)
        if not safe:
            issues.append(f"Safety violation: {reason}")
            return False, issues

        # 3. Structure check: must have TOOL_META
        if "TOOL_META" not in source:
            issues.append("Missing TOOL_META dictionary")

        # 4. Structure check: must have run() function
        if "def run(" not in source:
            issues.append("Missing run() function")

        # 5. Check for reasonable size
        if len(source) > 10000:
            issues.append("Script too large (>10000 chars)")

        # 6. Ensure TOOL_META has required fields
        if "TOOL_META" in source:
            if '"name"' not in source and "'name'" not in source:
                issues.append("TOOL_META missing 'name' field")
            if '"description"' not in source and "'description'" not in source:
                issues.append("TOOL_META missing 'description' field")

        is_valid = len(issues) == 0
        if is_valid:
            logger.debug("Tool script validation passed")
        else:
            logger.warning(f"Tool script validation failed: {issues}")

        return is_valid, issues

    def validate_file(self, script_path: str) -> tuple[bool, list[str]]:
        """Validate a tool script file."""
        from pathlib import Path

        path = Path(script_path)
        if not path.exists():
            return False, [f"File not found: {script_path}"]
        if not path.suffix == ".py":
            return False, [f"Not a Python file: {script_path}"]

        source = path.read_text(encoding="utf-8")
        return self.validate_script(source)
