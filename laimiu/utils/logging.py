"""Logging setup for Laimiu — quiet console, detailed file logs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from laimiu.constants import LAIMIU_HOME


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the Laimiu logger.

    Console: only ERROR and above (keeps chat clean).
    File: full detail at the configured level.
    """
    logger = logging.getLogger("laimiu")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler — only errors, no timestamps (clean chat)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

    # File handler — everything
    log_dir = LAIMIU_HOME / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "laimiu.log", encoding="utf-8")
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s")
    )
    logger.addHandler(file_handler)

    return logger
