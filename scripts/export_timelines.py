#!/usr/bin/env python3
"""Export exact keyframe timestamps and frame IDs for selected videos."""

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
    else:  # argparse prevents this; keep the function safe for direct use.
        raise ValueError("Specify --video-id or --all")
    return selected if limit is None else selected[:limit]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--video-id", help="Process one video, e.g. L21_V001")
    selection.add_argument("--all", action="store_true", help="Process every video")
    parser.add_argument("--limit", type=int, help="Limit selected videos for a smoke run")
    parser.add_argument("--output", help="Output JSONL path")
    parser.add_argument(
        "--strict", action="store_true", help="Require keyframe images and object JSON files"
    )
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
        else config.paths.artifacts_root / "timelines" / "keyframes.jsonl"
    )

    def records():
        for video in videos:
            yield from timeline_from_manifest(
                manifest, video.video_id, require_sidecars=args.strict
            )

    write_jsonl(records(), output)
    print(json.dumps({"output": str(output), "videos": len(videos)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
