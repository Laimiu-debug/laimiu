"""Laimiu CLI entry point.

Bootstrap order:
1. Environment check (Python version + missing packages auto-install)
2. Normal startup (Guardian → Config → Chat)
"""

import sys


def _bootstrap() -> None:
    """Phase 1: Check environment before importing any heavy modules."""
    # Inline Python version check (can't import anything yet)
    if sys.version_info < (3, 11):
        print(f"[ERROR] Python 3.11+ required (you have {sys.version_info.major}.{sys.version_info.minor})")
        print("  Download: https://www.python.org/downloads/")
        sys.exit(1)

    # Check & install missing packages
    from laimiu.bootstrap import ensure_environment
    if not ensure_environment():
        sys.exit(1)


def main() -> None:
    """Entry point."""
    _bootstrap()

    from laimiu.cli.app import main as _main
    _main()


if __name__ == "__main__":
    main()
