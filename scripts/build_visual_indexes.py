#!/usr/bin/env python3
"""Build keyframe and/or temporal-window visual indexes from supplied CLIP features."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.data.manifest import build_manifest
from agentforce.data.schemas import DatasetManifest
from agentforce.data.scope import (
    assert_exact_scope,
    normalize_video_ids,
    scope_hash,
    select_videos,
)
from agentforce.indexing.build_visual import build_visual_index
from agentforce.indexing.build_windows import build_visual_window_index, read_window_records
from agentforce.utils.io import atomic_write_json


def _manifest(config) -> DatasetManifest:
    path = config.paths.artifacts_root / "manifests" / "dataset.json"
    return DatasetManifest.read_json(path) if path.exists() else build_manifest(
        config.paths.dataset_root, validate=False
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--level", choices=("keyframe", "window", "both"), default="both"
    )
    parser.add_argument("--windows", help="Temporal-window JSONL path")
    parser.add_argument("--output-dir", help="Index output directory")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--video-ids",
        "--video-id",
        dest="video_ids",
        nargs="+",
        metavar="VIDEO_ID",
        help="Build an explicit subset inside the configured scope",
    )
    selection.add_argument(
        "--all", action="store_true", help="Build the complete configured scope"
    )
    parser.add_argument(
        "--limit", type=int, help="Window-index smoke run: limit the number of windows"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.limit is not None and args.level in {"keyframe", "both"}:
        raise ValueError("--limit only applies to --level window")

    config = load_config(project_path(args.config))
    manifest = _manifest(config)
    videos = select_videos(
        manifest,
        configured_ids=config.scope.video_ids,
        requested_ids=args.video_ids,
        all_configured=args.all,
    )
    selected_video_ids = tuple(sorted(video.video_id for video in videos))
    canonical_output_dir = config.paths.artifacts_root / "indexes"
    output_dir = project_path(args.output_dir) if args.output_dir else canonical_output_dir
    canonical_build = output_dir.resolve() == canonical_output_dir.resolve()
    if canonical_build:
        if args.limit is not None:
            raise ValueError("--limit requires an explicit non-canonical --output-dir")
        expected_ids = config.scope.video_ids or tuple(
            video.video_id for video in manifest.videos
        )
        assert_exact_scope(
            selected_video_ids,
            expected_ids,
            label="Canonical visual indexes",
        )

    expected_encoder = {
        "model": config.embeddings.clip_model,
        "pretrained": config.embeddings.clip_pretrained,
    }
    manifests: dict[str, object] = {}
    if args.level in {"keyframe", "both"}:
        manifests["keyframe"] = build_visual_index(
            config.paths.dataset_root,
            output_dir,
            dtype=args.dtype,
            video_ids=selected_video_ids,
            expected_encoder=expected_encoder,
        )

    if args.level in {"window", "both"}:
        windows_path = (
            project_path(args.windows)
            if args.windows
            else config.paths.artifacts_root / "windows" / "temporal_windows.jsonl"
        )
        windows = read_window_records(windows_path)
        input_video_ids = normalize_video_ids(
            str(row.get("video_id", "")) for row in windows
        )
        if canonical_build:
            assert_exact_scope(
                input_video_ids,
                selected_video_ids,
                label="Temporal windows used by canonical visual indexes",
            )
        selected_set = set(selected_video_ids)
        windows = [
            row
            for row in windows
            if str(row.get("video_id", "")).strip().upper() in selected_set
        ]
        present_ids = set(
            normalize_video_ids(str(row.get("video_id", "")) for row in windows)
        )
        missing_ids = sorted(selected_set - present_ids)
        if missing_ids:
            raise ValueError(
                "No temporal windows found for selected video IDs: "
                + ", ".join(missing_ids)
            )
        if args.limit is not None:
            windows = windows[: args.limit]
        window_manifest = build_visual_window_index(
            windows, config.paths.dataset_root, output_dir, dtype=args.dtype
        )
        indexed_video_ids = sorted(
            normalize_video_ids(str(row.get("video_id", "")) for row in windows)
        )
        window_manifest.update(
            {
                "video_ids": indexed_video_ids,
                "scope_hash": scope_hash(indexed_video_ids),
                "expected_encoder": expected_encoder,
            }
        )
        atomic_write_json(output_dir / "visual_windows.manifest.json", window_manifest)
        manifests["window"] = window_manifest

    print(json.dumps(manifests, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
