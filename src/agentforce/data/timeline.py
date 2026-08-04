"""Canonical keyframe timeline loaders and lookup helpers."""

from __future__ import annotations

from bisect import bisect_left
import csv
from pathlib import Path
from typing import Iterable, Sequence

from .layout import IMAGE_SUFFIXES
from .schemas import DatasetManifest, KeyframeRecord, ensure_unique_ids, write_jsonl


def keyframe_uid(video_id: str, keyframe_number: int) -> str:
    return f"{video_id}_K{keyframe_number:06d}"


def _numbered_files(directory: Path | None, suffixes: set[str]) -> dict[int, Path]:
    if directory is None or not directory.is_dir():
        return {}
    result: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in suffixes or not path.stem.isdigit():
            continue
        number = int(path.stem)
        if number in result:
            raise ValueError(f"Duplicate numbered file {number} in {directory}")
        result[number] = path
    return result


def read_keyframe_timeline(
    mapping_path: str | Path,
    video_id: str,
    *,
    image_dir: str | Path | None = None,
    object_dir: str | Path | None = None,
    require_sidecars: bool = False,
) -> list[KeyframeRecord]:
    """Read an organizer CSV and join numbered keyframe/object sidecars.

    ``visual_embedding_row`` is zero based while organizer keyframe numbers are
    one based. The explicit field prevents this common off-by-one bug from
    leaking into retrieval results.
    """

    mapping = Path(mapping_path)
    image_paths = _numbered_files(
        Path(image_dir) if image_dir is not None else None, IMAGE_SUFFIXES
    )
    object_paths = _numbered_files(Path(object_dir) if object_dir is not None else None, {".json"})
    records: list[KeyframeRecord] = []
    with mapping.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"n", "pts_time", "fps", "frame_idx"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Mapping {mapping} is missing columns: {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                number = int(row["n"])
                timestamp = float(row["pts_time"])
                fps = float(row["fps"])
                frame_idx = int(row["frame_idx"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid mapping value at {mapping}:{line_number}") from exc
            if number <= 0 or timestamp < 0 or fps <= 0 or frame_idx < 0:
                raise ValueError(f"Out-of-range mapping value at {mapping}:{line_number}")
            expected_number = len(records) + 1
            if number != expected_number:
                raise ValueError(
                    f"Expected sequential keyframe n={expected_number}, got {number} "
                    f"at {mapping}:{line_number}"
                )
            image = image_paths.get(number)
            objects = object_paths.get(number)
            if require_sidecars and image_dir is not None and image is None:
                raise FileNotFoundError(f"Missing keyframe image {number} for {video_id}")
            if require_sidecars and object_dir is not None and objects is None:
                raise FileNotFoundError(f"Missing object JSON {number} for {video_id}")
            records.append(
                KeyframeRecord(
                    keyframe_uid=keyframe_uid(video_id, number),
                    video_id=video_id,
                    keyframe_number=number,
                    frame_idx=frame_idx,
                    pts_time=timestamp,
                    fps=fps,
                    image_path=str(image) if image is not None else None,
                    object_path=str(objects) if objects is not None else None,
                    visual_embedding_row=number - 1,
                )
            )
    validate_timeline(records)
    return records


def timeline_from_manifest(
    manifest: DatasetManifest, video_id: str, *, require_sidecars: bool = False
) -> list[KeyframeRecord]:
    try:
        video = manifest.by_video_id()[video_id]
    except KeyError as exc:
        raise KeyError(f"Unknown video_id: {video_id}") from exc
    root = Path(manifest.dataset_root)
    mapping = video.resolve_path(root, "keyframe_map_path")
    if mapping is None:
        raise FileNotFoundError(f"No keyframe mapping for {video_id}")
    return read_keyframe_timeline(
        mapping,
        video_id,
        image_dir=video.resolve_path(root, "keyframe_dir"),
        object_dir=video.resolve_path(root, "object_dir"),
        require_sidecars=require_sidecars,
    )


def validate_timeline(records: Sequence[KeyframeRecord]) -> None:
    ensure_unique_ids(records, "keyframe_uid")
    previous: KeyframeRecord | None = None
    for record in records:
        if previous is not None:
            if record.video_id != previous.video_id:
                raise ValueError("A keyframe timeline may only contain one video_id")
            if record.keyframe_number <= previous.keyframe_number:
                raise ValueError("Keyframe numbers must be strictly increasing")
            if record.pts_time < previous.pts_time or record.frame_idx < previous.frame_idx:
                raise ValueError("Keyframe time/frame values must be monotonic")
        previous = record


def nearest_keyframe(
    records: Sequence[KeyframeRecord], timestamp: float
) -> KeyframeRecord | None:
    if not records:
        return None
    times = [record.pts_time for record in records]
    index = bisect_left(times, timestamp)
    if index == 0:
        return records[0]
    if index == len(records):
        return records[-1]
    before, after = records[index - 1], records[index]
    return before if timestamp - before.pts_time <= after.pts_time - timestamp else after


def keyframes_between(
    records: Sequence[KeyframeRecord], start: float, end: float
) -> list[KeyframeRecord]:
    if start < 0 or end < start:
        raise ValueError("Expected 0 <= start <= end")
    return [record for record in records if start <= record.pts_time < end]


def export_timeline(records: Iterable[KeyframeRecord], path: str | Path) -> None:
    write_jsonl(records, path)
