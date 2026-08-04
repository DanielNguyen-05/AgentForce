#!/usr/bin/env python3
"""Caption selected keyframes with Gemini using an on-disk response cache."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Sequence

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.data.manifest import build_manifest
from agentforce.data.schemas import DatasetManifest, VideoRecord, write_jsonl
from agentforce.data.timeline import timeline_from_manifest
from agentforce.gemini.cache import JsonFileCache
from agentforce.logging_utils import configure_logging
from agentforce.preprocessing.captions import CaptionProcessor, GeminiCaptionBackend


LOGGER = logging.getLogger("scripts.caption_keyframes")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--video-id", help="Caption one video, e.g. L21_V001")
    selection.add_argument("--all", action="store_true", help="Caption every video")
    parser.add_argument("--limit", type=int, help="Limit selected videos for a smoke run")
    parser.add_argument("--frame-limit", type=int, help="Limit captioned keyframes per video")
    parser.add_argument("--stride", type=int, default=3, help="Caption every Nth keyframe")
    parser.add_argument("--output-dir", help="Caption JSONL directory")
    parser.add_argument("--cache-dir", help="Gemini caption cache directory")
    parser.add_argument("--model", help="Override the Gemini model in the config")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--confirm-api-cost",
        action="store_true",
        help="Required with --all because this operation uses a paid API",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    if args.frame_limit is not None and args.frame_limit < 1:
        raise ValueError("--frame-limit must be positive")
    if args.all and not args.confirm_api_cost:
        raise ValueError("Captioning all videos can incur API cost; pass --confirm-api-cost")
    configure_logging(args.verbose)
    config = load_config(project_path(args.config))
    api_key = os.getenv(config.gemini.api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable {config.gemini.api_key_env} is not set")

    manifest = _manifest(config)
    videos = _select_videos(manifest, args.video_id, args.all, args.limit)
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else config.paths.artifacts_root / "captions"
    )
    cache_dir = (
        project_path(args.cache_dir)
        if args.cache_dir
        else config.paths.artifacts_root / "gemini" / "caption_cache"
    )
    processor = CaptionProcessor(
        GeminiCaptionBackend(model=args.model or config.gemini.model, api_key=api_key),
        cache=JsonFileCache(cache_dir),
    )

    completed = skipped = 0
    for video in videos:
        destination = output_dir / f"{video.video_id}.jsonl"
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue
        timeline = timeline_from_manifest(manifest, video.video_id)[:: args.stride]
        if args.frame_limit is not None:
            timeline = timeline[: args.frame_limit]
        LOGGER.info("Captioning %s (%d keyframes)", video.video_id, len(timeline))
        write_jsonl(
            (
                processor.process(record.image_path, record.keyframe_uid)
                for record in timeline
                if record.image_path is not None
            ),
            destination,
        )
        completed += 1

    print(
        json.dumps(
            {
                "completed": completed,
                "skipped": skipped,
                "output_dir": str(output_dir),
                "cache_dir": str(cache_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
