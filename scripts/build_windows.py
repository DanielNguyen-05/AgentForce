#!/usr/bin/env python3
"""Build overlapping multimodal temporal windows for selected videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.data.manifest import build_manifest
from agentforce.data.schemas import DatasetManifest, VideoRecord, write_jsonl
from agentforce.data.timeline import timeline_from_manifest
from agentforce.preprocessing.asr import read_transcript
from agentforce.preprocessing.windows import (
    WindowConfig,
    align_transcript_to_windows,
    attach_keyframe_text,
    attach_object_labels,
    build_temporal_windows,
)
from agentforce.utils.io import atomic_write_json, canonical_jsonl_sha256, file_sha256


def _manifest(config) -> DatasetManifest:
    path = config.paths.artifacts_root / "manifests" / "dataset.json"
    return DatasetManifest.read_json(path) if path.exists() else build_manifest(
        config.paths.dataset_root, validate=False
    )


def _select_videos(
    manifest: DatasetManifest, video_id: str | None, all_videos: bool, limit: int | None
) -> list[VideoRecord]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    if video_id:
        video = manifest.by_video_id().get(video_id.upper())
        if video is None:
            raise ValueError(f"Unknown video ID: {video_id}")
        selected = [video]
    elif all_videos:
        selected = list(manifest.videos)
    else:
        raise ValueError("Specify --video-id or --all")
    return selected if limit is None else selected[:limit]


def _load_text_mapping(path: Path, text_field: str) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "keyframe_uid" not in row:
                raise ValueError(f"Missing keyframe_uid at {path}:{line_number}")
            values[str(row["keyframe_uid"])] = str(row.get(text_field, ""))
    return values


def _load_object_labels(path: Path) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    if not path.exists():
        return values
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "keyframe_uid" not in row:
                raise ValueError(f"Missing keyframe_uid at {path}:{line_number}")
            counts = row.get("counts", {})
            if not isinstance(counts, dict):
                raise ValueError(f"Invalid object counts at {path}:{line_number}")
            values[str(row["keyframe_uid"])] = [str(label) for label in counts]
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--video-id", help="Build windows for one video")
    selection.add_argument("--all", action="store_true", help="Build windows for every video")
    parser.add_argument("--limit", type=int, help="Limit selected videos for a smoke run")
    parser.add_argument("--output", help="Output JSONL path")
    parser.add_argument("--length-seconds", type=float, help="Override window length")
    parser.add_argument("--stride-seconds", type=float, help="Override window stride")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    config = load_config(project_path(args.config))
    manifest = _manifest(config)
    videos = _select_videos(manifest, args.video_id, args.all, args.limit)
    output = (
        project_path(args.output)
        if args.output
        else config.paths.artifacts_root / "windows" / "temporal_windows.jsonl"
    )
    window_config = WindowConfig(
        length_seconds=(
            config.windows.length_seconds
            if args.length_seconds is None
            else args.length_seconds
        ),
        stride_seconds=(
            config.windows.stride_seconds
            if args.stride_seconds is None
            else args.stride_seconds
        ),
    )

    def generate():
        for video in videos:
            timeline = timeline_from_manifest(manifest, video.video_id)
            windows = build_temporal_windows(
                video.video_id,
                timeline,
                duration_seconds=video.duration_seconds,
                config=window_config,
            )
            transcript_path = (
                config.paths.artifacts_root / "transcripts" / f"{video.video_id}.json"
            )
            if transcript_path.exists():
                align_transcript_to_windows(windows, read_transcript(transcript_path).segments)

            ocr = _load_text_mapping(
                config.paths.artifacts_root / "ocr" / f"{video.video_id}.jsonl",
                "normalized_text",
            )
            if ocr:
                attach_keyframe_text(windows, ocr, target_field="ocr_text")

            captions = _load_text_mapping(
                config.paths.artifacts_root / "captions" / f"{video.video_id}.jsonl",
                "caption_text",
            )
            if captions:
                attach_keyframe_text(windows, captions, target_field="caption_text")

            objects = _load_object_labels(
                config.paths.artifacts_root / "objects" / f"{video.video_id}.jsonl"
            )
            if objects:
                attach_object_labels(windows, objects)
            yield from windows

    write_jsonl(generate(), output)
    manifest_path = output.with_suffix(".manifest.json")
    window_manifest = {
        "name": "temporal_windows",
        "path": str(output),
        "sha256": file_sha256(output),
        "canonical_records_hash": canonical_jsonl_sha256(
            output, exclude_fields=("metadata_text",)
        ),
        "video_count": len(videos),
        "video_ids": [video.video_id for video in videos],
        "window_length_seconds": window_config.length_seconds,
        "window_stride_seconds": window_config.stride_seconds,
        "dataset_manifest_created_at": manifest.created_at,
        "schema_version": manifest.schema_version,
    }
    atomic_write_json(manifest_path, window_manifest)
    print(
        json.dumps(
            {"output": str(output), "manifest": str(manifest_path), "videos": len(videos)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
