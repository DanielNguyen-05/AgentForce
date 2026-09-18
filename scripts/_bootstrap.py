"""Shared path setup for scripts executed directly from the repository.

Every executable file in this directory can be run with ``python scripts/...``
without installing the project first.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.toml"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agentforce.environment import load_project_env  # noqa: E402 - src path is set above

load_project_env(PROJECT_ROOT, override=False)


def project_path(value: str | Path) -> Path:
    """Resolve a user path consistently, relative to the repository root."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
