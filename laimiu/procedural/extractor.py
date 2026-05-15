"""Pattern extractor - detects automatable patterns from tracked tool calls."""

from __future__ import annotations

import logging
from typing import Any

from laimiu.procedural.tracker import Pattern, PatternTracker

logger = logging.getLogger("laimiu.procedural.extractor")


class PatternExtractor:
    """Analyzes tracked patterns and identifies candidates for tool generation.

    A pattern is considered extractable when:
    1. Its memory strength >= extract_strength threshold
    2. It has a reasonable success rate (>= 50%)
    3. The args suggest a consistent, automatable workflow
    """

    def __init__(self, tracker: PatternTracker, extract_strength: float = 0.6):
        self.tracker = tracker
        self.extract_strength = extract_strength

    def find_extractable_patterns(self) -> list[Pattern]:
        """Find patterns that should be converted to tools."""
        patterns = self.tracker.get_extractable_patterns()

        # Filter: only patterns with clear structure
        extractable = []
        for pattern in patterns:
            if self._is_automatable(pattern):
                extractable.append(pattern)
            else:
                logger.debug(
                    f"Pattern '{pattern.tool_name}' strength={pattern.strength:.2f} "
                    f"({pattern.level}) but doesn't look automatable"
                )

        return extractable

    def _is_automatable(self, pattern: Pattern) -> bool:
        """Heuristic: is this pattern suitable for automation?"""
        # Need examples to analyze
        if not pattern.examples:
            return False

        # Check that args have consistent structure
        if len(pattern.examples) >= 2:
            arg_keys_sets = [
                set(ex.get("args", {}).keys()) for ex in pattern.examples
            ]
            # All examples should share at least one arg key
            common_keys = set.intersection(*arg_keys_sets) if arg_keys_sets else set()
            if not common_keys:
                return False

        # Success rate should be decent
        if pattern.success_rate < 0.5:
            return False

        return True

    def generate_tool_prompt(self, pattern: Pattern) -> str:
        """Generate a prompt for the LLM to write a tool script.

        This prompt describes the repeated pattern and asks the LLM to
        write a Python script that automates it.
        """
        examples_text = ""
        for i, ex in enumerate(pattern.examples[:3], 1):
            args_str = ", ".join(f"{k}={v}" for k, v in ex.get("args", {}).items())
            examples_text += f"\n  Example {i}: {pattern.tool_name}({args_str}) -> {'success' if ex.get('success') else 'failure'}"

        return f"""You are writing a Python tool script for the Laimiu AI agent.

The agent has repeatedly used the '{pattern.tool_name}' tool in the same pattern ({pattern.occurrence} times, {pattern.success_rate:.0%} success rate, memory strength: {pattern.strength:.2f}/{pattern.level}).

Examples of how it's been called:
{examples_text}

Please write a Python script that automates this common workflow. The script must:

1. Define a `run()` function that takes relevant parameters
2. Include a `TOOL_META` dict with name, description, and parameters
3. Use only safe operations (no os.system, subprocess, eval, exec)
4. Be self-contained and deterministic

Format:
```python
\"\"\"
[Tool description - what it does and why it was created]
\"\"\"


def run(param1: str, param2: str = "default") -> str:
    \"\"\"[Function docstring]\"\"\"
    # Implementation
    return result

TOOL_META = {{
    "name": "tool_name",
    "description": "What this tool does",
    "parameters": {{
        "type": "object",
        "properties": {{
            "param1": {{"type": "string", "description": "..."}},
            "param2": {{"type": "string", "description": "...", "default": "default"}}
        }},
        "required": ["param1"]
    }}
}}
```"""
