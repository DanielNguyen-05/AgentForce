"""Deterministic on-disk cache for paid Gemini requests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_cache_key(payload: Mapping[str, Any], image_digests: Mapping[str, str] | None = None) -> str:
    """Hash every value that can change a model answer.

    Image *contents* are represented by SHA-256 digests, not paths or mtimes, so
    replacing a frame at the same path cannot return a stale answer.
    """

    envelope = {
        "payload": dict(payload),
        "image_digests": dict(sorted((image_digests or {}).items())),
    }
    return hashlib.sha256(_canonical_json(envelope)).hexdigest()


class JsonFileCache:
    """Simple process-safe-enough cache using atomic same-directory replaces."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _validate_key(key: str) -> None:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("Cache key must be a lowercase SHA-256 hex digest")

    def path_for(self, key: str) -> Path:
        self._validate_key(key)
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Gemini cache entry {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Gemini cache entry must be a JSON object: {path}")
        return value

    def set(self, key: str, value: Mapping[str, Any]) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = _canonical_json(dict(value)) + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return path
