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

# Patterns forbidden in code execution — only truly destructive operations
# Allow: __import__, exec, compile (needed for useful code execution)
# Block: system-level destruction
FORBIDDEN_CODE_PATTERNS = [
    "os.system(",
    "subprocess.call(",
    "subprocess.Popen(",
    "subprocess.run(",
    "rm -rf",
    "shutil.rmtree(",
    "os.remove(",
]


def is_command_dangerous(command: str) -> bool:
    """Check if a shell command is potentially dangerous."""
    cmd_lower = command.lower().strip()
    return any(pattern.lower() in cmd_lower for pattern in DANGEROUS_COMMANDS)


def is_code_safe(source: str) -> tuple[bool, str]:
    """Check if generated Python code is safe to execute.

    Returns (is_safe, reason).
    """
    # Decode base64 payloads before checking patterns
    import base64
    import re
    decoded_source = source
    b64_pattern = re.compile(r'base64\.b64decode\(["\']([A-Za-z0-9+/=]+)["\']\)')
    for match in b64_pattern.finditer(source):
        try:
            decoded = base64.b64decode(match.group(1)).decode("utf-8", errors="replace")
            decoded_source += "\n" + decoded
        except Exception:
            pass

    for pattern in FORBIDDEN_CODE_PATTERNS:
        if pattern in source or pattern in decoded_source:
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
    """Check if a path is a critical boot file that should not be modified.

    Only protects the minimum files needed for Laimiu to start.
    Everything else (tools, dream, memory, etc.) can be modified for self-evolution.
    """
    from pathlib import Path

    resolved = str(Path(path).resolve())

    import laimiu
    laimiu_src = str(Path(laimiu.__file__).parent.resolve())

    # Only protect critical boot files
    protected_files = {
        str(Path(laimiu_src) / "__init__.py"),
        str(Path(laimiu_src) / "__main__.py"),
        str(Path(laimiu_src) / "constants.py"),
        str(Path(laimiu_src) / "bootstrap.py"),
        str(Path(laimiu_src) / "config" / "settings.py"),
        str(Path(laimiu_src) / "cli" / "app.py"),
        str(Path(laimiu_src) / "cli" / "setup.py"),
        str(Path(laimiu_src) / "utils" / "safety.py"),
        str(Path(laimiu_src) / "utils" / "io.py"),
        str(Path(laimiu_src) / "utils" / "logging.py"),
    }

    if resolved in protected_files:
        return True

    # Protect pyproject.toml
    project_dir = str(Path(laimiu.__file__).parent.parent.resolve())
    if resolved == str(Path(project_dir) / "pyproject.toml"):
        return True

    return False
