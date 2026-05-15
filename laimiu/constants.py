"""Path constants and version info for Laimiu."""

from pathlib import Path

VERSION = "0.2.0"

# Runtime data directory
LAIMIU_HOME = Path.home() / ".laimiu"

# Sub-directories
MEMORY_DIR = LAIMIU_HOME / "memory"
NOTES_DIR = MEMORY_DIR / "notes"
TRANSCRIPTS_DIR = LAIMIU_HOME / "transcripts"
CHROMA_DIR = LAIMIU_HOME / "chroma"
TOOLS_DIR = LAIMIU_HOME / "tools"
ADAPTERS_DIR = LAIMIU_HOME / "adapters"
DREAM_DIR = LAIMIU_HOME / "dream"

# Snapshots directory
SNAPSHOTS_DIR = LAIMIU_HOME / "snapshots"

# Key files
CONFIG_FILE = LAIMIU_HOME / "config.yaml"
SOUL_FILE = LAIMIU_HOME / "SOUL.md"
MEMORY_INDEX_FILE = MEMORY_DIR / "MEMORY.md"
USER_PREFS_FILE = MEMORY_DIR / "user.md"
DREAM_STATE_FILE = DREAM_DIR / ".state.json"
DREAM_LOG_FILE = DREAM_DIR / ".dream_log.jsonl"
PATTERNS_FILE = DREAM_DIR / ".patterns.jsonl"
ENV_FILE = LAIMIU_HOME / ".env"

# Limits
MEMORY_INDEX_MAX_LINES = 200
RECALL_MAX_CHARS = 2000
VECTOR_SEARCH_RESULTS = 5
SYSTEM_PROMPT_MAX_TOKENS = 2000


def ensure_dirs() -> None:
    """Create all required directories if they don't exist."""
    for d in [
        LAIMIU_HOME,
        MEMORY_DIR,
        NOTES_DIR,
        TRANSCRIPTS_DIR,
        CHROMA_DIR,
        TOOLS_DIR,
        ADAPTERS_DIR,
        DREAM_DIR,
        SNAPSHOTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
