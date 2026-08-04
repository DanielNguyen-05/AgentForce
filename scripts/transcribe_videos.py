#!/usr/bin/env python3
"""Transcribe selected videos with Faster-Whisper and word timestamps."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Sequence

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.data.manifest import build_manifest
from agentforce.data.schemas import DatasetManifest, VideoRecord
from agentforce.logging_utils import configure_logging
from agentforce.preprocessing.asr import (
    ASRConfig as WhisperConfig,
    FasterWhisperAdapter,
    write_transcript,
)


LOGGER = logging.getLogger("scripts.transcribe_videos")


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
    selection.add_argument("--video-id", help="Transcribe one video, e.g. L21_V001")
    selection.add_argument("--all", action="store_true", help="Transcribe every video")
    parser.add_argument("--limit", type=int, help="Limit selected videos for a smoke run")
    parser.add_argument("--output-dir", help="Transcript directory")
    parser.add_argument("--model", help="Override the Whisper model in the config")
    parser.add_argument("--device", help="Override the inference device")
    parser.add_argument("--compute-type", help="Override the Whisper compute type")
    parser.add_argument(
        "--language",
        help="Language code; use 'auto' for language detection",
    )
    parser.add_argument(
        "--word-timestamps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable word timestamps",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    configure_logging(args.verbose)
    config = load_config(project_path(args.config))
    manifest = _manifest(config)
    videos = _select_videos(manifest, args.video_id, args.all, args.limit)
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else config.paths.artifacts_root / "transcripts"
    )
    language = config.asr.language if args.language is None else (
        None if args.language.casefold() == "auto" else args.language
    )
    word_timestamps = (
        config.asr.word_timestamps if args.word_timestamps is None else args.word_timestamps
    )
    adapter = FasterWhisperAdapter(
        WhisperConfig(
            model_name=args.model or config.asr.model,
            device=args.device or config.asr.device,
            compute_type=args.compute_type or config.asr.compute_type,
            language=language,
            word_timestamps=word_timestamps,
        )
    )

    completed = skipped = missing = 0
    for video in videos:
        destination = output_dir / f"{video.video_id}.json"
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue
        source = video.resolve_path(manifest.dataset_root, "video_path")
        if source is None or not source.is_file():
            LOGGER.warning("Skipping missing video %s", video.video_id)
            missing += 1
            continue
        LOGGER.info("Transcribing %s from %s", video.video_id, source)
        document = adapter.transcribe_video(source, video_id=video.video_id)
        write_transcript(document, destination)
        completed += 1

    print(
        json.dumps(
            {
                "completed": completed,
                "skipped": skipped,
                "missing": missing,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
