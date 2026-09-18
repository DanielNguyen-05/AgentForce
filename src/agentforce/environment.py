"""Project environment loading shared by every direct Python entry file."""

from __future__ import annotations

from pathlib import Path


def load_project_env(project_root: str | Path, *, override: bool = False) -> bool:
    """Load ``<project_root>/.env`` without replacing exported shell values.

    The helper deliberately receives an explicit root so running a script from
    another working directory still reads the repository's environment file.
    """

    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - dependency installation error
        raise RuntimeError(
            "Automatic .env loading requires python-dotenv. Install the project "
            "dependencies with `python -m pip install -e .`."
        ) from exc
    return bool(load_dotenv(Path(project_root) / ".env", override=override))


__all__ = ["load_project_env"]
