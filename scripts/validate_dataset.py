#!/usr/bin/env python3
"""Build and validate the canonical dataset manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.data.manifest import build_manifest
from agentforce.data.scope import assert_exact_scope, filter_manifest, select_videos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="TOML config path (default: configs/default.toml)",
    )
    parser.add_argument("--dataset-root", help="Override paths.dataset_root from the config")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--video-ids",
        "--video-id",
        dest="video_ids",
        nargs="+",
        metavar="VIDEO_ID",
        help="Validate an explicit subset inside the configured scope",
    )
    selection.add_argument(
        "--all", action="store_true", help="Validate the complete configured scope"
    )
    parser.add_argument("--output", help="Manifest JSON path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 if validation finds an error",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    config = load_config(project_path(args.config))
    dataset_root = (
        project_path(args.dataset_root) if args.dataset_root else config.paths.dataset_root
    )
    canonical_output = config.paths.artifacts_root / "manifests" / "dataset.json"
    output = project_path(args.output) if args.output else canonical_output

    manifest = build_manifest(dataset_root, validate=True)
    videos = select_videos(
        manifest,
        configured_ids=config.scope.video_ids,
        requested_ids=args.video_ids,
        all_configured=args.all,
    )
    selected_video_ids = tuple(video.video_id for video in videos)
    if output.resolve() == canonical_output.resolve():
        expected_ids = config.scope.video_ids or tuple(
            video.video_id for video in manifest.videos
        )
        assert_exact_scope(
            selected_video_ids,
            expected_ids,
            label="Canonical dataset manifest",
        )
    manifest = filter_manifest(manifest, selected_video_ids)
    manifest.write_json(output)
    summary = {
        "manifest": str(output),
        "dataset_root": str(dataset_root),
        "videos": len(manifest.videos),
        "video_ids": [video.video_id for video in manifest.videos],
        "errors": manifest.error_count,
        "warnings": manifest.warning_count,
        "missing_video_ids": [
            video.video_id for video in manifest.videos if "video" in video.missing_components
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if args.strict and manifest.error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
