"""Logging setup for Laimiu."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from laimiu.constants import LAIMIU_HOME


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the Laimiu logger."""
    logger = logging.getLogger("laimiu")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)

    # File handler
    log_dir = LAIMIU_HOME / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "laimiu.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s")
    )
    logger.addHandler(file_handler)

    return logger
