"""Atomic file I/O utilities for safe writes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write content to a file atomically using write-to-temp-then-rename.

    This prevents partial/corrupt files if the process crashes mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    dir_name = str(path.parent)

    # Write to a temp file in the same directory (ensures same filesystem for rename)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".laimiu_tmp_")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        # On Windows, need to remove destination before rename
        if os.name == "nt" and path.exists():
            path.unlink()
        os.replace(tmp_path, str(path))
    except BaseException:
        # Clean up temp file on any error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
