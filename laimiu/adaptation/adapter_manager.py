"""Adapter manager - version management for LoRA adapters."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from laimiu.constants import ADAPTERS_DIR

logger = logging.getLogger("laimiu.adaptation.adapter_manager")


class AdapterManager:
    """Manages LoRA adapter versions for model personalization.

    Handles:
    - Versioning adapter snapshots
    - Rolling back to previous versions
    - Tracking adapter metadata
    """

    def __init__(self, adapters_dir: Path | None = None):
        self.adapters_dir = adapters_dir or ADAPTERS_DIR
        self.adapters_dir.mkdir(parents=True, exist_ok=True)

    def create_version(self, adapter_path: str, metadata: dict[str, Any] | None = None) -> str:
        """Register a new adapter version.

        Returns the version identifier.
        """
        # Find next version number
        existing = sorted(
            [d.name for d in self.adapters_dir.iterdir() if d.is_dir()]
        )
        if existing:
            last_num = int(existing[-1].lstrip("v"))
            version = f"v{last_num + 1:03d}"
        else:
            version = "v001"

        version_dir = self.adapters_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        meta = {
            "version": version,
            "created": datetime.now().isoformat(),
            "source_path": adapter_path,
            **(metadata or {}),
        }
        meta_path = version_dir / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(f"Created adapter version: {version}")
        return version

    def list_versions(self) -> list[dict[str, Any]]:
        """List all adapter versions."""
        versions = []
        for version_dir in sorted(self.adapters_dir.iterdir()):
            if not version_dir.is_dir():
                continue
            meta_path = version_dir / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    versions.append(meta)
                except (json.JSONDecodeError, OSError):
                    versions.append({"version": version_dir.name, "error": "corrupt metadata"})
            else:
                versions.append({"version": version_dir.name})
        return versions

    def get_version(self, version: str) -> dict[str, Any] | None:
        """Get metadata for a specific version."""
        meta_path = self.adapters_dir / version / "metadata.json"
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def get_latest(self) -> str | None:
        """Get the latest adapter version identifier."""
        versions = sorted([d.name for d in self.adapters_dir.iterdir() if d.is_dir()])
        return versions[-1] if versions else None
