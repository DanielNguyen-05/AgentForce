"""Build a canonical memory-mapped visual index from supplied CLIP shards."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from agentforce.data.scope import scope_hash


def _normalize_video_ids(video_ids: Collection[str]) -> tuple[str, ...]:
    if isinstance(video_ids, (str, bytes)):
        raise TypeError("video_ids must be a collection of video IDs, not a string")

    normalized: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for raw_video_id in video_ids:
        if not isinstance(raw_video_id, str):
            raise TypeError("Every video ID must be a string")
        video_id = raw_video_id.strip().upper()
        if not video_id:
            raise ValueError("Video IDs must not be empty")
        if video_id in seen:
            duplicates.add(video_id)
        else:
            seen.add(video_id)
            normalized.append(video_id)
    if duplicates:
        raise ValueError(f"Duplicate video IDs: {', '.join(sorted(duplicates))}")
    if not normalized:
        raise ValueError("video_ids must contain at least one video ID")
    return tuple(sorted(normalized))


def _feature_files(
    dataset_root: Path, video_ids: Collection[str] | None = None
) -> tuple[list[Path], tuple[str, ...]]:
    available: dict[str, Path] = {}
    duplicate_files: set[str] = set()
    for path in sorted((dataset_root / "clip-features-32").glob("*.npy")):
        video_id = path.stem.upper()
        if video_id in available:
            duplicate_files.add(video_id)
        else:
            available[video_id] = path
    if duplicate_files:
        raise ValueError(
            "Duplicate feature shards for video IDs: "
            + ", ".join(sorted(duplicate_files))
        )

    selected_ids = (
        tuple(sorted(available))
        if video_ids is None
        else _normalize_video_ids(video_ids)
    )
    if not selected_ids:
        raise FileNotFoundError(
            f"No .npy features found under {dataset_root / 'clip-features-32'}"
        )
    missing = [video_id for video_id in selected_ids if video_id not in available]
    if missing:
        raise FileNotFoundError(
            "Missing CLIP feature shards for video IDs: " + ", ".join(missing)
        )
    return [available[video_id] for video_id in selected_ids], selected_ids


def _temporary_path(directory: Path, *, prefix: str, suffix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{prefix}.",
        suffix=suffix,
    )
    os.close(descriptor)
    return Path(raw_path)


def _mapping_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _inspect_feature_files(paths: Sequence[Path]) -> tuple[int, int]:
    total = 0
    dimension: int | None = None
    for path in paths:
        matrix = np.load(path, mmap_mode="r")
        if matrix.ndim != 2:
            raise ValueError(f"Feature shard is not 2D: {path} {matrix.shape}")
        if dimension is None:
            dimension = int(matrix.shape[1])
        elif dimension != matrix.shape[1]:
            raise ValueError(
                f"Mixed feature dimensions: {path} has {matrix.shape[1]}, expected {dimension}"
            )
        total += int(matrix.shape[0])
    if dimension is None:
        raise FileNotFoundError("No visual feature shards were selected")
    return total, dimension


def inspect_visual_shards(
    dataset_root: str | Path,
    video_ids: Collection[str] | None = None,
) -> tuple[int, int]:
    root = Path(dataset_root)
    feature_paths, _ = _feature_files(root, video_ids)
    return _inspect_feature_files(feature_paths)


def build_visual_index(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    dtype: str = "float16",
    video_ids: Collection[str] | None = None,
    expected_encoder: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Concatenate existing CLIP shards and write row-aligned JSONL metadata."""

    root = Path(dataset_root).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    feature_paths, selected_video_ids = _feature_files(root, video_ids)
    total, dimension = _inspect_feature_files(feature_paths)
    selected_scope_hash = scope_hash(selected_video_ids)
    requested_dtype = np.dtype(dtype)
    if requested_dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("Visual index dtype must be float16 or float32")

    vectors_path = destination / "visual_keyframes.npy"
    metadata_path = destination / "visual_keyframes.jsonl"
    manifest_path = destination / "visual_keyframes.manifest.json"
    temp_vectors = _temporary_path(
        destination, prefix="visual_keyframes", suffix=".npy.tmp"
    )
    temp_metadata = _temporary_path(
        destination, prefix="visual_keyframes", suffix=".jsonl.tmp"
    )
    temp_manifest = _temporary_path(
        destination, prefix="visual_keyframes", suffix=".manifest.json.tmp"
    )

    try:
        matrix = np.lib.format.open_memmap(
            temp_vectors, mode="w+", dtype=requested_dtype, shape=(total, dimension)
        )
        row_offset = 0
        with temp_metadata.open("w", encoding="utf-8") as metadata_handle:
            for feature_path in feature_paths:
                video_id = feature_path.stem.upper()
                features = np.load(feature_path, mmap_mode="r")
                mapping_path = root / "map-keyframes" / f"{video_id}.csv"
                if not mapping_path.is_file():
                    raise FileNotFoundError(
                        f"Missing keyframe mapping for video ID {video_id}: {mapping_path}"
                    )
                mappings = _mapping_rows(mapping_path)
                if len(mappings) != features.shape[0]:
                    raise ValueError(
                        f"{video_id}: {features.shape[0]} feature rows but "
                        f"{len(mappings)} mapping rows"
                    )
                next_offset = row_offset + features.shape[0]
                matrix[row_offset:next_offset] = features.astype(
                    requested_dtype, copy=False
                )
                for local_row, mapping in enumerate(mappings):
                    number = int(mapping["n"])
                    record = {
                        "vector_id": f"{video_id}_K{number:06d}",
                        "keyframe_uid": f"{video_id}_K{number:06d}",
                        "video_id": video_id,
                        "keyframe_number": number,
                        "frame_idx": int(mapping["frame_idx"]),
                        "pts_time": float(mapping["pts_time"]),
                        "fps": float(mapping["fps"]),
                        "source_feature": str(feature_path.relative_to(root)),
                        "source_row": local_row,
                    }
                    metadata_handle.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                row_offset = next_offset
        matrix.flush()
        del matrix

        source_digest = hashlib.sha256()
        source_digest.update(f"scope:{selected_scope_hash}\n".encode())
        for feature_path in feature_paths:
            video_id = feature_path.stem.upper()
            mapping_path = root / "map-keyframes" / f"{video_id}.csv"
            for source_path in (feature_path, mapping_path):
                stat = source_path.stat()
                relative_path = source_path.relative_to(root).as_posix()
                source_digest.update(
                    f"{relative_path}:{stat.st_size}:{stat.st_mtime_ns}\n".encode()
                )
        manifest = {
            "name": "visual_keyframes",
            "vectors_path": vectors_path.name,
            "metadata_path": metadata_path.name,
            "row_count": total,
            "dimension": dimension,
            "dtype": requested_dtype.name,
            "normalized": True,
            "source": "dataset/clip-features-32",
            "video_ids": list(selected_video_ids),
            "scope_hash": selected_scope_hash,
            "source_signature": source_digest.hexdigest(),
            "expected_encoder": (
                dict(expected_encoder) if expected_encoder is not None else None
            ),
            "created_at": datetime.now(UTC).isoformat(),
            "builder_version": 1,
        }
        temp_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temp_vectors, vectors_path)
        os.replace(temp_metadata, metadata_path)
        os.replace(temp_manifest, manifest_path)
    finally:
        for temporary_path in (temp_vectors, temp_metadata, temp_manifest):
            temporary_path.unlink(missing_ok=True)
    return manifest
