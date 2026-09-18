"""Build and validate a canonical manifest without mutating the dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Iterable

from .layout import DatasetLayout, collection_from_video_id, is_video_id
from .schemas import DatasetManifest, ValidationIssue, VideoRecord


def _index_paths(
    paths: Iterable[Path],
    identifier: Callable[[Path], str],
    component: str,
) -> tuple[dict[str, Path], list[ValidationIssue]]:
    index: dict[str, Path] = {}
    issues: list[ValidationIssue] = []
    for path in paths:
        video_id = identifier(path).upper()
        if video_id in index:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="duplicate_component",
                    message=f"Multiple {component} entries: {index[video_id]} and {path}",
                    video_id=video_id,
                    path=str(path),
                )
            )
            continue
        index[video_id] = path
    return index, issues


def _direct_files(directory: Path, suffix: str) -> list[Path]:
    if not directory.exists():
        return []
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() == suffix and is_video_id(path.stem)
    ]


def _count_files(
    directory: Path | None, suffix: str | set[str] | None = None
) -> int | None:
    if directory is None or not directory.is_dir():
        return None
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file()
        and (
            suffix is None
            or path.suffix.lower() == suffix
            or isinstance(suffix, set) and path.suffix.lower() in suffix
        )
    )


def _mapping_summary(path: Path | None) -> tuple[int | None, float | None, float | None]:
    if path is None or not path.is_file():
        return None, None, None
    count = 0
    fps: float | None = None
    last_timestamp: float | None = None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                count += 1
                if fps is None:
                    fps = float(row["fps"])
                last_timestamp = float(row["pts_time"])
    except (KeyError, TypeError, ValueError, csv.Error):
        return None, None, None
    return count, fps, last_timestamp


def _feature_shape(path: Path | None) -> tuple[int | None, int | None, str | None]:
    if path is None or not path.is_file():
        return None, None, None
    try:
        import numpy as np

        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2:
            return int(array.shape[0]) if array.ndim else 0, None, str(array.dtype)
        return int(array.shape[0]), int(array.shape[1]), str(array.dtype)
    except (ImportError, OSError, ValueError, EOFError):
        return None, None, None


def _media_summary(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def build_manifest(dataset_root: str | Path, *, validate: bool = True) -> DatasetManifest:
    """Discover all logical video IDs and their associated dataset components."""

    layout = DatasetLayout.from_path(dataset_root)
    video_index, issues = _index_paths(layout.iter_video_files(), lambda path: path.stem, "videos")
    keyframe_index, component_issues = _index_paths(
        layout.iter_keyframe_dirs(), lambda path: path.name, "keyframes"
    )
    issues.extend(component_issues)
    map_index, component_issues = _index_paths(
        _direct_files(layout.keyframe_maps, ".csv"), lambda path: path.stem, "keyframe maps"
    )
    issues.extend(component_issues)
    feature_index, component_issues = _index_paths(
        _direct_files(layout.clip_features, ".npy"), lambda path: path.stem, "CLIP features"
    )
    issues.extend(component_issues)
    object_index, component_issues = _index_paths(
        layout.iter_object_dirs(), lambda path: path.name, "object directories"
    )
    issues.extend(component_issues)
    media_index, component_issues = _index_paths(
        _direct_files(layout.media_info, ".json"), lambda path: path.stem, "media info"
    )
    issues.extend(component_issues)

    video_ids = sorted(
        set(video_index)
        | set(keyframe_index)
        | set(map_index)
        | set(feature_index)
        | set(object_index)
        | set(media_index)
    )
    records: list[VideoRecord] = []
    for video_id in video_ids:
        video_path = video_index.get(video_id)
        keyframe_dir = keyframe_index.get(video_id)
        map_path = map_index.get(video_id)
        feature_path = feature_index.get(video_id)
        object_dir = object_index.get(video_id)
        media_path = media_index.get(video_id)

        mapping_count, mapping_fps, last_timestamp = _mapping_summary(map_path)
        feature_count, feature_dimension, feature_dtype = _feature_shape(feature_path)
        media = _media_summary(media_path)
        duration = _optional_float(media.get("length"))
        if duration is None and last_timestamp is not None:
            duration = last_timestamp

        components = {
            "video": video_path,
            "keyframes": keyframe_dir,
            "keyframe_map": map_path,
            "clip_features": feature_path,
            "objects": object_dir,
            "media_info": media_path,
        }
        missing = [name for name, path in components.items() if path is None]
        split = video_path.parent.name if video_path is not None else (
            keyframe_dir.parent.name if keyframe_dir is not None else None
        )
        records.append(
            VideoRecord(
                video_id=video_id,
                collection=collection_from_video_id(video_id),
                split=split,
                video_path=layout.relative(video_path),
                keyframe_dir=layout.relative(keyframe_dir),
                keyframe_map_path=layout.relative(map_path),
                clip_feature_path=layout.relative(feature_path),
                object_dir=layout.relative(object_dir),
                media_info_path=layout.relative(media_path),
                keyframe_count=_count_files(keyframe_dir, {".jpg", ".jpeg", ".png", ".webp"}),
                mapping_count=mapping_count,
                object_frame_count=_count_files(object_dir, ".json"),
                feature_count=feature_count,
                feature_dimension=feature_dimension,
                feature_dtype=feature_dtype,
                fps=mapping_fps,
                duration_seconds=duration,
                frame_count=(round(duration * mapping_fps) if duration and mapping_fps else None),
                title=_optional_string(media.get("title")),
                description=_optional_string(media.get("description")),
                author=_optional_string(media.get("author")),
                missing_components=missing,
            )
        )

    manifest = DatasetManifest(dataset_root=str(layout.root), videos=records, issues=issues)
    if validate:
        manifest.issues.extend(validate_manifest(manifest))
    return manifest


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def validate_manifest(manifest: DatasetManifest) -> list[ValidationIssue]:
    """Validate presence, cardinality and keyframe timeline invariants."""

    root = Path(manifest.dataset_root)
    issues: list[ValidationIssue] = []
    for video in manifest.videos:
        for component in video.missing_components:
            severity = "error" if component in {
                "video", "keyframes", "keyframe_map", "clip_features"
            } else "warning"
            issues.append(
                ValidationIssue(
                    severity=severity,
                    code="missing_component",
                    message=f"Missing dataset component: {component}",
                    video_id=video.video_id,
                )
            )

        counts = {
            "keyframes": video.keyframe_count,
            "mapping": video.mapping_count,
            "objects": video.object_frame_count,
            "features": video.feature_count,
        }
        known_counts = {name: count for name, count in counts.items() if count is not None}
        if len(set(known_counts.values())) > 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="component_count_mismatch",
                    message="Component row counts differ: "
                    + ", ".join(f"{name}={count}" for name, count in known_counts.items()),
                    video_id=video.video_id,
                )
            )
        if video.feature_dimension is not None and video.feature_dimension <= 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_feature_dimension",
                    message=f"Feature dimension must be positive, got {video.feature_dimension}",
                    video_id=video.video_id,
                    path=video.clip_feature_path,
                )
            )
        if video.feature_dtype is not None and video.feature_dtype not in {"float16", "float32"}:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="unexpected_feature_dtype",
                    message=f"Expected float16/float32 CLIP vectors, got {video.feature_dtype}",
                    video_id=video.video_id,
                    path=video.clip_feature_path,
                )
            )
        if video.clip_feature_path is not None and video.feature_count is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="unreadable_feature_file",
                    message="Could not read CLIP feature shape",
                    video_id=video.video_id,
                    path=video.clip_feature_path,
                )
            )

        mapping = video.resolve_path(root, "keyframe_map_path")
        if mapping is not None:
            issues.extend(_validate_mapping(mapping, video.video_id, video.duration_seconds))
    return issues


def _validate_mapping(
    path: Path, video_id: str, duration_seconds: float | None
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    previous_n = 0
    previous_time = -1.0
    previous_frame = -1
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"n", "pts_time", "fps", "frame_idx"}
            if not required.issubset(reader.fieldnames or []):
                return [
                    ValidationIssue(
                        severity="error",
                        code="invalid_mapping_header",
                        message=f"Expected columns {sorted(required)}, got {reader.fieldnames}",
                        video_id=video_id,
                        path=str(path),
                    )
                ]
            for line_number, row in enumerate(reader, start=2):
                number = int(row["n"])
                timestamp = float(row["pts_time"])
                fps = float(row["fps"])
                frame_idx = int(row["frame_idx"])
                if number <= previous_n or timestamp < previous_time or frame_idx < previous_frame:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="non_monotonic_mapping",
                            message=f"Timeline is not monotonic at line {line_number}",
                            video_id=video_id,
                            path=str(path),
                        )
                    )
                    break
                if fps <= 0 or timestamp < 0 or frame_idx < 0:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="invalid_mapping_value",
                            message=f"Invalid time/fps/frame at line {line_number}",
                            video_id=video_id,
                            path=str(path),
                        )
                    )
                    break
                if abs(timestamp * fps - frame_idx) > max(2.0, fps * 0.1):
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            code="timestamp_frame_drift",
                            message=f"Timestamp/frame disagreement at line {line_number}",
                            video_id=video_id,
                            path=str(path),
                        )
                    )
                previous_n, previous_time, previous_frame = number, timestamp, frame_idx
    except (OSError, csv.Error, KeyError, TypeError, ValueError) as exc:
        issues.append(
            ValidationIssue(
                severity="error",
                code="unreadable_mapping",
                message=str(exc),
                video_id=video_id,
                path=str(path),
            )
        )
    if duration_seconds is not None and previous_time > duration_seconds + 2.0:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="mapping_exceeds_duration",
                message=(
                    f"Last keyframe {previous_time:.3f}s exceeds "
                    f"duration {duration_seconds:.3f}s"
                ),
                video_id=video_id,
                path=str(path),
            )
        )
    return issues
