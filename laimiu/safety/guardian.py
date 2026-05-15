"""Guardian - self-healing system with snapshots and health checks."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from laimiu.constants import (
    CHROMA_DIR,
    CONFIG_FILE,
    DREAM_DIR,
    LAIMIU_HOME,
    MEMORY_DIR,
    MEMORY_INDEX_FILE,
    SOUL_FILE,
    TOOLS_DIR,
    TRANSCRIPTS_DIR,
    SNAPSHOTS_DIR,
)
from laimiu.utils.io import atomic_write

logger = logging.getLogger("laimiu.safety.guardian")

MAX_SNAPSHOTS = 5


class SnapshotManager:
    """Manages snapshots of the ~/.laimiu/ state for rollback."""

    def __init__(self, snapshots_dir: Path | None = None):
        self.snapshots_dir = snapshots_dir or SNAPSHOTS_DIR
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Files/dirs to include in snapshots (key state only)
    _INCLUDE_PATTERNS = [
        "config.yaml",
        "SOUL.md",
        "memory/MEMORY.md",
        "memory/user.md",
        "dream/.state.json",
        "dream/.patterns.jsonl",
    ]

    # Directories to include (recursively, but not huge ones)
    _INCLUDE_DIRS = [
        "memory/notes",
        "tools",
    ]

    def create_snapshot(self, tag: str | None = None) -> str:
        """Create a tar.gz snapshot of critical state files.

        Returns the tag name of the created snapshot.
        """
        if tag is None:
            tag = datetime.now().strftime("snap_%Y%m%d_%H%M%S")

        snapshot_path = self.snapshots_dir / f"{tag}.tar.gz"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        base_dir = LAIMIU_HOME
        included_files: list[tuple[Path, str]] = []

        # Collect individual files
        for pattern in self._INCLUDE_PATTERNS:
            fpath = base_dir / pattern
            if fpath.exists():
                included_files.append((fpath, pattern))

        # Collect directories
        for dirname in self._INCLUDE_DIRS:
            dirpath = base_dir / dirname
            if dirpath.exists():
                for f in dirpath.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(base_dir)
                        included_files.append((f, str(rel)))

        if not included_files:
            logger.warning("No files to snapshot")
            return tag

        with tarfile.open(str(snapshot_path), "w:gz") as tar:
            for fpath, arcname in included_files:
                tar.add(str(fpath), arcname=arcname)

        logger.info(f"Created snapshot: {tag} ({snapshot_path.stat().st_size} bytes)")

        # Auto-cleanup old snapshots
        self._cleanup_old_snapshots()

        return tag

    def restore_snapshot(self, tag: str) -> bool:
        """Restore a snapshot by tag name. Returns True on success."""
        snapshot_path = self.snapshots_dir / f"{tag}.tar.gz"
        if not snapshot_path.exists():
            logger.error(f"Snapshot not found: {tag}")
            return False

        try:
            with tarfile.open(str(snapshot_path), "r:gz") as tar:
                tar.extractall(path=str(LAIMIU_HOME), filter="data")
            logger.info(f"Restored snapshot: {tag}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore snapshot {tag}: {e}")
            return False

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List all available snapshots with metadata."""
        snapshots = []
        for f in sorted(self.snapshots_dir.glob("*.tar.gz")):
            stat = f.stat()
            tag = f.stem
            snapshots.append({
                "tag": tag,
                "file": str(f),
                "size_bytes": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        return snapshots

    def get_latest_snapshot_tag(self) -> str | None:
        """Get the tag of the most recent snapshot."""
        snapshots = self.list_snapshots()
        if not snapshots:
            return None
        return snapshots[-1]["tag"]

    def _cleanup_old_snapshots(self) -> None:
        """Keep only the most recent MAX_SNAPSHOTS snapshots."""
        snapshots = self.list_snapshots()
        if len(snapshots) <= MAX_SNAPSHOTS:
            return
        # Remove oldest
        for snap in snapshots[:-MAX_SNAPSHOTS]:
            try:
                Path(snap["file"]).unlink()
                logger.info(f"Removed old snapshot: {snap['tag']}")
            except OSError as e:
                logger.warning(f"Failed to remove snapshot {snap['tag']}: {e}")


class HealthChecker:
    """Checks the health of all Laimiu subsystems."""

    def check_all(self) -> tuple[bool, list[str]]:
        """Run all health checks. Returns (healthy, issues_list)."""
        issues: list[str] = []
        issues.extend(self.check_config())
        issues.extend(self.check_memory())
        issues.extend(self.check_tools())
        issues.extend(self.check_soul())
        issues.extend(self.check_providers())
        return len(issues) == 0, issues

    def check_config(self) -> list[str]:
        """Check if config.yaml is parseable."""
        issues = []
        if not CONFIG_FILE.exists():
            issues.append("config.yaml does not exist")
            return issues
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                issues.append("config.yaml is not a valid YAML mapping")
        except yaml.YAMLError:
            issues.append("config.yaml has invalid YAML syntax")
        except Exception as e:
            issues.append(f"config.yaml read error: {e}")
        return issues

    def check_memory(self) -> list[str]:
        """Check memory subsystem health."""
        issues = []
        if not MEMORY_INDEX_FILE.exists():
            issues.append("MEMORY.md does not exist (will be auto-created)")
        elif MEMORY_INDEX_FILE.stat().st_size == 0:
            issues.append("MEMORY.md is empty")
        # ChromaDB check - just verify directory exists
        if not CHROMA_DIR.exists():
            issues.append("chroma/ directory does not exist")
        return issues

    def check_tools(self) -> list[str]:
        """Check if generated tools can be imported (syntax check only, no execution)."""
        issues = []
        if not TOOLS_DIR.exists():
            return issues
        for f in TOOLS_DIR.glob("*.py"):
            if f.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(f.stem, str(f))
                if spec is None:
                    issues.append(f"Tool {f.name}: cannot create import spec")
            except Exception as e:
                issues.append(f"Tool {f.name}: {e}")
        return issues

    def check_soul(self) -> list[str]:
        """Check if SOUL.md exists."""
        if not SOUL_FILE.exists():
            return ["SOUL.md does not exist"]
        return []

    def check_providers(self) -> list[str]:
        """Check if at least one provider is configured."""
        if not CONFIG_FILE.exists():
            return ["No config file - no providers configured"]
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            models = data.get("provider", {}).get("models", {})
            if not models:
                return ["No providers configured"]
            return []
        except Exception:
            return ["Cannot read provider configuration"]


class Guardian:
    """Self-healing system that protects Laimiu's state.

    On startup, runs health checks. If unhealthy, attempts to restore
    from the latest snapshot. If no snapshots exist, rebuilds defaults.
    """

    def __init__(self):
        self.snapshot_mgr = SnapshotManager()
        self.health = HealthChecker()

    def startup_check(self) -> bool:
        """Run startup health check and auto-recover if needed.

        Returns True if system is healthy (or recovered), False if
        running in safe mode.
        """
        healthy, issues = self.health.check_all()

        if healthy:
            logger.info("Startup health check passed")
            return True

        logger.warning(f"Health check found {len(issues)} issue(s): {issues}")

        # Try to restore from snapshot
        latest = self.snapshot_mgr.get_latest_snapshot_tag()
        if latest:
            logger.info(f"Attempting to restore from snapshot: {latest}")
            if self.snapshot_mgr.restore_snapshot(latest):
                # Re-check after restore
                healthy_after, issues_after = self.health.check_all()
                if healthy_after:
                    logger.info("Successfully recovered from snapshot")
                    return True
                logger.warning(f"Still unhealthy after restore: {issues_after}")

        # Fall back to safe mode boot
        self.safe_mode_boot(issues)
        return False

    def pre_mutation_snapshot(self) -> str | None:
        """Create a snapshot before a mutation (Dream, tool generation, etc.).

        Returns snapshot tag or None on failure.
        """
        try:
            return self.snapshot_mgr.create_snapshot()
        except Exception as e:
            logger.error(f"Failed to create pre-mutation snapshot: {e}")
            return None

    def safe_mode_boot(self, issues: list[str]) -> None:
        """Minimal startup: rebuild default configuration files."""
        logger.warning("Entering safe mode - rebuilding defaults")

        # Ensure directories exist
        for d in [LAIMIU_HOME, MEMORY_DIR, TOOLS_DIR, DREAM_DIR, TRANSCRIPTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        # Rebuild config.yaml if broken
        if any("config" in i for i in issues):
            from laimiu.config.settings import LaimiuConfig, ProviderConfig, ProviderModelConfig

            config = LaimiuConfig()
            # Try to salvage provider settings from env
            import os
            models = {}
            if os.environ.get("DEEPSEEK_API_KEY"):
                models["deepseek"] = ProviderModelConfig(
                    base_url="https://api.deepseek.com",
                    model="deepseek-v4-pro",
                    api_key=os.environ["DEEPSEEK_API_KEY"],
                )
                config.provider.default = "deepseek"
            elif os.environ.get("OPENAI_API_KEY"):
                models["glm"] = ProviderModelConfig(
                    base_url="https://open.bigmodel.cn/api/coding/paas/v4",
                    model="glm-4.7-flashx",
                    api_key=os.environ["OPENAI_API_KEY"],
                )
                config.provider.default = "glm"
            else:
                models["ollama"] = ProviderModelConfig(
                    base_url="http://localhost:11434/v1",
                    model="llama3.1",
                    api_key="not-needed",
                )
                config.provider.default = "ollama"
            config.provider.models = models
            config.save(CONFIG_FILE)

        # Rebuild SOUL.md if missing
        if any("SOUL" in i for i in issues):
            content = """# Laimiu

You are Laimiu, a personal AI assistant that learns and evolves with your user.

## Personality
- Direct, helpful, and proactive
- Remember user preferences and apply them
- When uncertain, ask rather than guess
- Use tools when they can help accomplish tasks

## Behavior
- Always respond in the same language the user uses
- Be concise but thorough
- Use memory_recall to check relevant past conversations when needed
- Track your own learning and improvement
"""
            atomic_write(SOUL_FILE, content)

        # Rebuild MEMORY.md if missing
        if any("MEMORY" in i for i in issues):
            content = "# Laimiu Memory Index\n# Auto-managed. Do not edit manually.\n\n"
            atomic_write(MEMORY_INDEX_FILE, content)

        # Take a snapshot of the rebuilt state
        self.snapshot_mgr.create_snapshot("safe_mode_rebuild")

        logger.info("Safe mode rebuild complete")
