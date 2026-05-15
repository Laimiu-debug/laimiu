"""Safety utilities for Laimiu."""

from __future__ import annotations


# Commands that require user approval
DANGEROUS_COMMANDS = [
    "rm -rf",
    "rm -r",
    "rmdir",
    "del /",
    "format",
    "mkfs",
    "dd if=",
    ":(){ :|:& };:",
    "> /dev/sd",
    "chmod -R 777",
    "chown",
    "shutdown",
    "reboot",
    "init 0",
    "init 6",
    "systemctl stop",
    "service stop",
    "pip uninstall",
    "npm uninstall",
    "git push --force",
    "git reset --hard",
    "git clean -f",
    "DROP TABLE",
    "DROP DATABASE",
    "DELETE FROM",
    "TRUNCATE",
]

# Patterns forbidden in generated tools
FORBIDDEN_CODE_PATTERNS = [
    "os.system(",
    "subprocess.call(",
    "subprocess.Popen(",
    "__import__(",
    "eval(",
    "exec(",
    "compile(",
    "rm -rf",
    "os.remove(",
    "shutil.rmtree(",
]


def is_command_dangerous(command: str) -> bool:
    """Check if a shell command is potentially dangerous."""
    cmd_lower = command.lower().strip()
    return any(pattern.lower() in cmd_lower for pattern in DANGEROUS_COMMANDS)


def is_code_safe(source: str) -> tuple[bool, str]:
    """Check if generated Python code is safe to execute.

    Returns (is_safe, reason).
    """
    for pattern in FORBIDDEN_CODE_PATTERNS:
        if pattern in source:
            return False, f"Forbidden pattern found: {pattern}"
    return True, "OK"


def sanitize_path(path: str, allowed_roots: list[str] | None = None) -> bool:
    """Check if a file path is safe to access (no path traversal)."""
    from pathlib import Path

    resolved = Path(path).resolve()
    # Block path traversal
    if ".." in str(path):
        return False
    # Block system directories
    system_dirs = ["/etc", "/sys", "/proc", "/dev", "C:\\Windows", "C:\\System32"]
    for sd in system_dirs:
        if str(resolved).startswith(sd):
            return False
    return True


def is_source_write_protected(path: str) -> bool:
    """Check if a path is inside Laimiu's own source directory — should not be modified at runtime.

    Only protects: laimiu/**/*.py and pyproject.toml.
    Everything else (memory.md, ~/.laimiu/, etc.) is fair game.
    """
    from pathlib import Path

    resolved = str(Path(path).resolve())

    # Protect the laimiu package source directory (*.py files only)
    import laimiu
    laimiu_src = str(Path(laimiu.__file__).parent.resolve())
    if resolved.startswith(laimiu_src) and resolved.endswith(".py"):
        return True

    # Protect the project root pyproject.toml
    project_dir = str(Path(laimiu.__file__).parent.parent.resolve())
    if resolved == str(Path(project_dir) / "pyproject.toml"):
        return True

    return False
