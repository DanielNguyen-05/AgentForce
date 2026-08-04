"""Build a canonical memory-mapped visual index from supplied CLIP shards."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
import csv
import json
import os
from datetime import datetime, timezone
import hashlib

import numpy as np


def _feature_files(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "clip-features-32").glob("*.npy"))


def _mapping_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def inspect_visual_shards(dataset_root: str | Path) -> tuple[int, int]:
    root = Path(dataset_root)
    total = 0
    dimension: int | None = None
    for path in _feature_files(root):
        matrix = np.load(path, mmap_mode="r")
        if matrix.ndim != 2:
            raise ValueError(f"Feature shard is not 2D: {path} {matrix.shape}")
        if dimension is None:
            dimension = int(matrix.shape[1])
        elif dimension != matrix.shape[1]:
            raise ValueError(f"Mixed feature dimensions: {path} has {matrix.shape[1]}, expected {dimension}")
        total += int(matrix.shape[0])
    if dimension is None:
        raise FileNotFoundError(f"No .npy features found under {root / 'clip-features-32'}")
    return total, dimension


def build_visual_index(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    dtype: str = "float16",
) -> dict[str, object]:
    """Concatenate existing CLIP shards and write row-aligned JSONL metadata."""

    root = Path(dataset_root).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    total, dimension = inspect_visual_shards(root)
    requested_dtype = np.dtype(dtype)
    if requested_dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("Visual index dtype must be float16 or float32")

    vectors_path = destination / "visual_keyframes.npy"
    metadata_path = destination / "visual_keyframes.jsonl"
    manifest_path = destination / "visual_keyframes.manifest.json"
    temp_vectors = destination / ".visual_keyframes.npy.tmp"
    temp_metadata = destination / ".visual_keyframes.jsonl.tmp"

    matrix = np.lib.format.open_memmap(
        temp_vectors, mode="w+", dtype=requested_dtype, shape=(total, dimension)
    )
    row_offset = 0
    with temp_metadata.open("w", encoding="utf-8") as metadata_handle:
        for feature_path in _feature_files(root):
            video_id = feature_path.stem
            features = np.load(feature_path, mmap_mode="r")
            mapping_path = root / "map-keyframes" / f"{video_id}.csv"
            mappings = _mapping_rows(mapping_path)
            if len(mappings) != features.shape[0]:
                raise ValueError(
                    f"{video_id}: {features.shape[0]} feature rows but {len(mappings)} mapping rows"
                )
            next_offset = row_offset + features.shape[0]
            matrix[row_offset:next_offset] = features.astype(requested_dtype, copy=False)
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
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            row_offset = next_offset
    matrix.flush()
    del matrix
    os.replace(temp_vectors, vectors_path)
    os.replace(temp_metadata, metadata_path)
    source_digest = hashlib.sha256()
    for path in _feature_files(root):
        stat = path.stat()
        source_digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    manifest = {
        "name": "visual_keyframes",
        "vectors_path": vectors_path.name,
        "metadata_path": metadata_path.name,
        "row_count": total,
        "dimension": dimension,
        "dtype": requested_dtype.name,
        "normalized": True,
        "source": "dataset/clip-features-32",
        "source_signature": source_digest.hexdigest(),
        "expected_encoder": {"model": "ViT-B-32", "pretrained": "openai"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "builder_version": 1,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
