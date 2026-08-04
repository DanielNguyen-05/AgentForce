#!/usr/bin/env python3
"""Transcribe the configured video scope with a local PhoWhisper CT2 model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Sequence

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.data.scope import load_configured_manifest, select_videos
from agentforce.logging_utils import configure_logging
from agentforce.preprocessing.asr import (
    ASRConfig as WhisperConfig,
    FasterWhisperAdapter,
    TranscriptSchemaError,
    read_transcript,
    write_transcript,
)


LOGGER = logging.getLogger("scripts.transcribe_videos")


def _model_reference(raw: str) -> str:
    """Resolve explicit project-local model paths while preserving Hugging Face IDs."""

    value = Path(raw).expanduser()
    if value.is_absolute():
        return str(value)
    if raw.startswith((".", "models/")):
        return str(project_path(value))
    local = project_path(value)
    return str(local) if local.exists() else raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--video-id",
        action="append",
        dest="single_video_ids",
        help="Transcribe one scoped video; repeat to select several",
    )
    selection.add_argument(
        "--video-ids",
        nargs="+",
        metavar="VIDEO_ID",
        help="Explicit scoped video IDs (default: [scope].video_ids)",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Process the complete configured scope",
    )
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
    parser.add_argument(
        "--rewrite-legacy",
        action="store_true",
        help="Validate and rewrite existing legacy PhoWhisper JSON to canonical schema",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first video error instead of reporting remaining videos",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    configure_logging(args.verbose)
    config = load_config(project_path(args.config))
    manifest = load_configured_manifest(config)
    requested_ids = args.video_ids or args.single_video_ids
    videos = select_videos(
        manifest,
        configured_ids=config.scope.video_ids,
        requested_ids=requested_ids,
        all_configured=args.all,
        limit=args.limit,
    )
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else config.paths.outputs_root / "transcripts"
    )
    language = config.asr.language if args.language is None else (
        None if args.language.casefold() == "auto" else args.language
    )
    word_timestamps = (
        config.asr.word_timestamps if args.word_timestamps is None else args.word_timestamps
    )
    model_reference = _model_reference(args.model or config.asr.model)
    model_path = Path(model_reference)
    needs_inference = args.overwrite or any(
        not (output_dir / f"{video.video_id}.json").is_file() for video in videos
    )
    if needs_inference and model_path.is_absolute() and not model_path.is_dir():
        raise FileNotFoundError(
            f"Local PhoWhisper model not found: {model_path}. Run `python "
            "convert_phowhisper.py` first, or pass a Hugging Face model ID to --model."
        )
    adapter = FasterWhisperAdapter(
        WhisperConfig(
            model_name=model_reference,
            device=args.device or config.asr.device,
            compute_type=args.compute_type or config.asr.compute_type,
            language=language,
            beam_size=config.asr.beam_size,
            vad_filter=config.asr.vad_filter,
            vad_min_silence_duration_ms=config.asr.vad_min_silence_duration_ms,
            word_timestamps=word_timestamps,
            condition_on_previous_text=config.asr.condition_on_previous_text,
        )
    )

    completed = skipped = missing = 0
    failures: list[dict[str, str]] = []
    for video in videos:
        destination = output_dir / f"{video.video_id}.json"
        if destination.exists() and not args.overwrite:
            try:
                existing = read_transcript(
                    destination,
                    rewrite_legacy=args.rewrite_legacy,
                )
                if existing.video_id.upper() != video.video_id:
                    raise TranscriptSchemaError(
                        f"Transcript video_id={existing.video_id!r} does not match "
                        f"{video.video_id!r}"
                    )
            except (OSError, ValueError) as exc:
                message = (
                    f"Existing transcript is invalid: {destination}: {exc}. "
                    "Use --overwrite to regenerate it."
                )
                if args.fail_fast:
                    raise ValueError(message) from exc
                LOGGER.error(message)
                failures.append({"video_id": video.video_id, "error": message})
                continue
            skipped += 1
            continue
        source = video.resolve_path(manifest.dataset_root, "video_path")
        if source is None or not source.is_file():
            LOGGER.warning("Skipping missing video %s", video.video_id)
            missing += 1
            continue
        LOGGER.info("Transcribing %s from %s", video.video_id, source)
        try:
            document = adapter.transcribe_video(source, video_id=video.video_id)
            write_transcript(document, destination)
            completed += 1
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # Keep a three-video batch resumable.
            if args.fail_fast:
                raise
            LOGGER.exception("Transcription failed for %s", video.video_id)
            failures.append({"video_id": video.video_id, "error": str(exc)})

    print(
        json.dumps(
            {
                "completed": completed,
                "skipped": skipped,
                "missing": missing,
                "failed": len(failures),
                "failures": failures,
                "video_ids": [video.video_id for video in videos],
                "model": adapter.config.model_name,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return 1 if failures or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
