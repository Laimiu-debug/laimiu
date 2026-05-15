"""Bootstrap environment check — runs before anything else.

Checks Python version and required packages.
Auto-installs missing dependencies via pip.
"""

from __future__ import annotations

import subprocess
import sys


# Required packages with minimum versions
REQUIRED_PACKAGES = {
    "openai": "1.30.0",
    "pydantic": "2.5.0",
    "yaml": None,          # pyyaml imports as `yaml`
    "rich": "13.0.0",
    "prompt_toolkit": "3.0.0",
    "chromadb": "0.5.0",
}

PIP_NAMES = {
    "yaml": "pyyaml>=6.0",
    "openai": "openai>=1.30.0",
    "pydantic": "pydantic>=2.5.0",
    "rich": "rich>=13.0.0",
    "prompt_toolkit": "prompt_toolkit>=3.0.0",
    "chromadb": "chromadb>=0.5.0",
}


def _check_python_version() -> bool:
    """Check Python >= 3.11."""
    if sys.version_info >= (3, 11):
        return True
    print(f"[ERROR] Python 3.11+ is required (you have {sys.version_info.major}.{sys.version_info.minor})")
    print("  Download: https://www.python.org/downloads/")
    return False


def _find_missing_packages() -> list[tuple[str, str]]:
    """Return list of (import_name, pip_spec) for missing packages."""
    missing = []
    for import_name, min_ver in REQUIRED_PACKAGES.items():
        try:
            mod = __import__(import_name)
            if min_ver and hasattr(mod, "__version__"):
                from packaging.version import Version
                if Version(mod.__version__) < Version(min_ver):
                    raise ImportError(f"too old: {mod.__version__} < {min_ver}")
        except ImportError:
            pip_spec = PIP_NAMES.get(import_name, import_name)
            missing.append((import_name, pip_spec))
    return missing


def _install_packages(pip_specs: list[str]) -> bool:
    """Install packages via pip. Returns True on success."""
    print(f"  Installing {len(pip_specs)} package(s)...")
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + pip_specs
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return True
        print(f"  pip error: {result.stderr.strip()}")
        return False
    except subprocess.TimeoutExpired:
        print("  pip install timed out (5 min)")
        return False
    except Exception as e:
        print(f"  pip install failed: {e}")
        return False


def ensure_environment() -> bool:
    """Check and fix the runtime environment.

    Returns True if environment is ready, False if fatal.
    """
    # 1. Python version
    if not _check_python_version():
        return False

    # 2. Missing packages
    missing = _find_missing_packages()
    if not missing:
        return True

    print()
    print(f"  Missing {len(missing)} required package(s):")
    for name, _ in missing:
        print(f"    - {name}")
    print()

    # Try auto-install
    pip_specs = [spec for _, spec in missing]
    print("  Attempting auto-install...")
    if _install_packages(pip_specs):
        print("  All packages installed successfully!")
        return True

    print()
    print("  Auto-install failed. Please install manually:")
    print(f"    pip install {' '.join(pip_specs)}")
    return False
