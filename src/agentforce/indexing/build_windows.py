"""Create row-aligned visual and text indexes for canonical temporal windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping
import csv
import json
import os
import re
import hashlib
from datetime import datetime, timezone

import numpy as np

from agentforce.embeddings.encoders import TextEncoder, l2_normalize
from .build_text import build_text_index


_KEYFRAME_UID = re.compile(r"^(?P<video>L\d+_V\d+)_K(?P<number>\d+)$", re.IGNORECASE)


def _window_records_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        canonical = {key: value for key, value in row.items() if key != "metadata_text"}
        digest.update(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def read_window_records(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("window_id"):
                raise ValueError(f"Invalid temporal window at {path}:{line_number}")
            rows.append(row)
    return rows


def _source_rows(window: Mapping[str, Any]) -> tuple[str, list[int]]:
    video_id = str(window["video_id"])
    rows: list[int] = []
    for uid in window.get("keyframe_uids", []):
        match = _KEYFRAME_UID.fullmatch(str(uid))
        if match is None or match.group("video").upper() != video_id.upper():
            raise ValueError(f"Invalid keyframe UID {uid!r} in window {window['window_id']}")
        rows.append(int(match.group("number")) - 1)
    return video_id, rows


def build_visual_window_index(
    windows: Iterable[Mapping[str, Any]],
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    dtype: str = "float16",
) -> dict[str, Any]:
    """Mean-pool supplied CLIP keyframes inside every non-empty window."""

    rows = [dict(row) for row in windows if row.get("keyframe_uids")]
    if not rows:
        raise ValueError("No windows contain keyframes")
    root = Path(dataset_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    first_video, _ = _source_rows(rows[0])
    first = np.load(root / "clip-features-32" / f"{first_video}.npy", mmap_mode="r")
    dimension = int(first.shape[1])
    vectors_path = output / "visual_windows.npy"
    metadata_path = output / "visual_windows.jsonl"
    manifest_path = output / "visual_windows.manifest.json"
    temporary_vectors = output / ".visual_windows.npy.tmp"
    temporary_metadata = output / ".visual_windows.jsonl.tmp"

    output_matrix = np.lib.format.open_memmap(
        temporary_vectors,
        mode="w+",
        dtype=np.dtype(dtype),
        shape=(len(rows), dimension),
    )
    feature_cache: dict[str, np.ndarray] = {}
    mapping_cache: dict[str, list[dict[str, str]]] = {}
    with temporary_metadata.open("w", encoding="utf-8") as metadata_handle:
        for output_row, window in enumerate(rows):
            video_id, source_rows = _source_rows(window)
            features = feature_cache.get(video_id)
            if features is None:
                features = np.load(
                    root / "clip-features-32" / f"{video_id}.npy", mmap_mode="r"
                )
                if features.ndim != 2 or features.shape[1] != dimension:
                    raise ValueError(f"Unexpected CLIP feature shape for {video_id}: {features.shape}")
                feature_cache[video_id] = features
            if min(source_rows) < 0 or max(source_rows) >= features.shape[0]:
                raise IndexError(f"Keyframe row outside feature shard for {window['window_id']}")
            pooled = np.asarray(features[source_rows], dtype=np.float32).mean(axis=0)
            pooled = l2_normalize(pooled)[0]
            output_matrix[output_row] = pooled.astype(dtype, copy=False)

            # Pick the keyframe most representative of the pooled window. This
            # gives KIS/Q&A a valid organizer frame ID without decoding video.
            block = np.asarray(features[source_rows], dtype=np.float32)
            representative_local = int(np.argmax(block @ pooled))
            representative_row = source_rows[representative_local]
            mappings = mapping_cache.get(video_id)
            if mappings is None:
                mapping_path = root / "map-keyframes" / f"{video_id}.csv"
                with mapping_path.open(newline="", encoding="utf-8-sig") as handle:
                    mappings = list(csv.DictReader(handle))
                if len(mappings) != features.shape[0]:
                    raise ValueError(f"Mapping/feature count mismatch for {video_id}")
                mapping_cache[video_id] = mappings
            mapping = mappings[representative_row]
            metadata = {
                key: value
                for key, value in window.items()
                if key not in {"asr_text", "ocr_text", "metadata_text"}
            }
            metadata["vector_id"] = str(window["window_id"])
            metadata["representative_keyframe_uid"] = f"{video_id}_K{int(mapping['n']):06d}"
            metadata["frame_idx"] = int(mapping["frame_idx"])
            metadata["pts_time"] = float(mapping["pts_time"])
            metadata["fps"] = float(mapping["fps"])
            metadata_handle.write(
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    output_matrix.flush()
    del output_matrix
    os.replace(temporary_vectors, vectors_path)
    os.replace(temporary_metadata, metadata_path)
    manifest = {
        "name": "visual_windows",
        "vectors_path": vectors_path.name,
        "metadata_path": metadata_path.name,
        "row_count": len(rows),
        "dimension": dimension,
        "dtype": np.dtype(dtype).name,
        "normalized": True,
        "pooling": "mean",
        "expected_encoder": {"model": "ViT-B-32", "pretrained": "openai"},
        "window_records_hash": _window_records_hash(rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "builder_version": 1,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_window_text_indexes(
    windows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    encoder: TextEncoder,
    *,
    fields: tuple[str, ...] = (
        "asr_text",
        "ocr_text",
        "object_labels",
        "metadata_text",
    ),
    batch_size: int = 128,
) -> dict[str, dict[str, Any]]:
    """Build independent modality indexes sharing each window ID."""

    rows = [dict(row) for row in windows]
    window_records_hash = _window_records_hash(rows)
    manifests: dict[str, dict[str, Any]] = {}
    for field in fields:
        def field_text(row: Mapping[str, Any]) -> str:
            value = row.get(field, "")
            if isinstance(value, (list, tuple, set)):
                return ", ".join(str(item) for item in value if str(item).strip())
            return str(value)

        source = (
            {
                "vector_id": row["window_id"],
                "window_id": row["window_id"],
                "video_id": row["video_id"],
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "start_frame": row.get("start_frame"),
                "end_frame": row.get("end_frame"),
                "representative_keyframe_uid": row.get("representative_keyframe_uid"),
                "frame_idx": row.get("frame_idx"),
                "pts_time": row.get("pts_time"),
                "fps": row.get("fps"),
                "text": field_text(row),
                field: row.get(field, ""),
                "source_field": field,
            }
            for row in rows
        )
        if any(field_text(row).strip() for row in rows):
            modality = "objects" if field == "object_labels" else field.removesuffix("_text")
            manifest = build_text_index(
                source,
                output_dir,
                f"{modality}_windows",
                encoder,
                batch_size=batch_size,
            )
            manifest["window_records_hash"] = window_records_hash
            manifest_path = Path(output_dir) / f"{modality}_windows.manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            manifests[modality] = manifest
    return manifests
