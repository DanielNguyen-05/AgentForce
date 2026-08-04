"""Dataset-scope helpers shared by preprocessing and runtime scripts.

The raw AIC dataset can contain hundreds of videos while an experiment may
intentionally target a much smaller, explicit subset.  These helpers make that
subset part of the data contract instead of relying on lexicographic
``--limit`` behavior.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
from typing import TYPE_CHECKING

from .manifest import build_manifest
from .schemas import DatasetManifest, VideoRecord

if TYPE_CHECKING:  # Avoid importing configuration during package initialization.
    from agentforce.config import AppConfig


def normalize_video_ids(video_ids: Iterable[str]) -> tuple[str, ...]:
    """Normalize, de-duplicate, and validate a sequence of video IDs."""

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in video_ids:
        video_id = str(raw).strip().upper()
        if not video_id:
            raise ValueError("Video IDs must not be empty")
        if video_id in seen:
            continue
        seen.add(video_id)
        normalized.append(video_id)
    return tuple(normalized)


def scope_hash(video_ids: Iterable[str]) -> str:
    """Return a stable fingerprint for an unordered video scope."""

    digest = hashlib.sha256()
    for video_id in sorted(normalize_video_ids(video_ids)):
        digest.update(video_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def filter_manifest(
    manifest: DatasetManifest,
    video_ids: Sequence[str],
    *,
    require_all: bool = True,
) -> DatasetManifest:
    """Return a manifest containing exactly ``video_ids`` in requested order."""

    requested = normalize_video_ids(video_ids)
    if not requested:
        return manifest
    by_id = manifest.by_video_id()
    missing = [video_id for video_id in requested if video_id not in by_id]
    if missing and require_all:
        raise ValueError(f"Unknown video IDs in configured scope: {', '.join(missing)}")
    selected = [by_id[video_id] for video_id in requested if video_id in by_id]
    selected_ids = {video.video_id for video in selected}
    issues = [
        issue
        for issue in manifest.issues
        if issue.video_id is None or issue.video_id in selected_ids
    ]
    return DatasetManifest(
        dataset_root=manifest.dataset_root,
        videos=selected,
        issues=issues,
        created_at=manifest.created_at,
        schema_version=manifest.schema_version,
    )


def load_configured_manifest(config: "AppConfig") -> DatasetManifest:
    """Load the persisted manifest and enforce the configured experiment scope."""

    path = config.paths.artifacts_root / "manifests" / "dataset.json"
    manifest = (
        DatasetManifest.read_json(path)
        if path.is_file()
        else build_manifest(config.paths.dataset_root, validate=False)
    )
    return filter_manifest(manifest, config.scope.video_ids)


def select_videos(
    manifest: DatasetManifest,
    *,
    configured_ids: Sequence[str],
    requested_ids: Sequence[str] | None = None,
    all_configured: bool = False,
    limit: int | None = None,
) -> list[VideoRecord]:
    """Resolve a script selection without ever escaping the configured scope.

    With neither ``requested_ids`` nor ``all_configured``, the complete
    configured scope is selected.  This makes the default command safe for a
    small experiment while ``--all`` can remain an explicit opt-in when the
    configured scope itself is empty (meaning the complete dataset).
    """

    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    allowed_manifest = filter_manifest(manifest, configured_ids)
    allowed = allowed_manifest.by_video_id()
    requested = normalize_video_ids(requested_ids or ())
    if requested:
        outside = [video_id for video_id in requested if video_id not in allowed]
        if outside:
            raise ValueError(
                "Requested video IDs are outside the configured scope: "
                + ", ".join(outside)
            )
        selected = [allowed[video_id] for video_id in requested]
    elif configured_ids:
        selected = list(allowed_manifest.videos)
    elif all_configured:
        selected = list(manifest.videos)
    else:
        raise ValueError("Specify --video-ids or --all when [scope].video_ids is empty")
    return selected if limit is None else selected[:limit]


def assert_exact_scope(video_ids: Iterable[str], expected: Iterable[str], *, label: str) -> None:
    """Fail when a canonical artifact was built for a different video set."""

    actual_set = set(normalize_video_ids(video_ids))
    expected_set = set(normalize_video_ids(expected))
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(f"{label} does not match configured scope ({', '.join(details)})")


__all__ = [
    "assert_exact_scope",
    "filter_manifest",
    "load_configured_manifest",
    "normalize_video_ids",
    "scope_hash",
    "select_videos",
]
