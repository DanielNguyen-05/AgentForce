#!/usr/bin/env python3
"""Export exact keyframe timestamps and frame IDs for selected videos."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.data.manifest import build_manifest
from agentforce.data.schemas import DatasetManifest, write_jsonl
from agentforce.data.scope import assert_exact_scope, select_videos
from agentforce.data.timeline import timeline_from_manifest


def _manifest(config) -> DatasetManifest:
    path = config.paths.artifacts_root / "manifests" / "dataset.json"
    return DatasetManifest.read_json(path) if path.exists() else build_manifest(
        config.paths.dataset_root, validate=False
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--video-ids",
        "--video-id",
        dest="video_ids",
        nargs="+",
        metavar="VIDEO_ID",
        help="Process an explicit subset inside the configured scope",
    )
    selection.add_argument(
        "--all", action="store_true", help="Process the complete configured scope"
    )
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
    videos = select_videos(
        manifest,
        configured_ids=config.scope.video_ids,
        requested_ids=args.video_ids,
        all_configured=args.all,
        limit=args.limit,
    )
    canonical_output = config.paths.artifacts_root / "timelines" / "keyframes.jsonl"
    output = project_path(args.output) if args.output else canonical_output
    if output.resolve() == canonical_output.resolve():
        if args.limit is not None:
            raise ValueError("--limit requires an explicit non-canonical --output")
        expected_ids = config.scope.video_ids or tuple(
            video.video_id for video in manifest.videos
        )
        assert_exact_scope(
            (video.video_id for video in videos),
            expected_ids,
            label="Canonical keyframe timeline",
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
