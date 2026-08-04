"""Shared path setup for scripts executed directly from the repository.

Every executable file in this directory can be run with ``python scripts/...``
without installing the project first.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.toml"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def project_path(value: str | Path) -> Path:
    """Resolve a user path consistently, relative to the repository root."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
