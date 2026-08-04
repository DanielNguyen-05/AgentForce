#!/usr/bin/env python3
"""Run PhoWhisper on one short video clip without touching canonical transcripts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Callable, Sequence

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.data.schemas import TranscriptDocument
from agentforce.data.scope import load_configured_manifest, select_videos
from agentforce.logging_utils import configure_logging
from agentforce.preprocessing.asr import ASRConfig, FasterWhisperAdapter
from agentforce.utils.io import atomic_write_json, utc_now


LOGGER = logging.getLogger("scripts.smoke_phowhisper")
MAX_SMOKE_DURATION_SECONDS = 60.0


def _model_reference(raw: str) -> str:
    """Resolve an explicit local model path while preserving Hub model IDs."""

    value = Path(raw).expanduser()
    if value.is_absolute():
        return str(value.resolve())
    if raw.startswith((".", "models/")):
        return str(project_path(value))
    local = project_path(value)
    return str(local) if local.exists() else raw


def _time_token(value: float) -> str:
    rendered = f"{value:.3f}".rstrip("0").rstrip(".") or "0"
    return rendered.replace(".", "p")


def default_output_path(
    outputs_root: Path,
    *,
    video_id: str,
    start_seconds: float,
    duration_seconds: float,
) -> Path:
    filename = (
        f"{video_id}_start{_time_token(start_seconds)}_duration{_time_token(duration_seconds)}.json"
    )
    return outputs_root / "smoke" / "phowhisper" / filename


def validate_output_path(
    output: str | Path,
    *,
    outputs_root: str | Path,
    overwrite: bool,
) -> Path:
    """Require a JSON destination below ``outputs/smoke``.

    Resolving both paths also prevents an existing symlink below the smoke
    directory from redirecting a result into any canonical artifact directory.
    """

    destination = Path(output).expanduser().resolve()
    smoke_root = (Path(outputs_root).expanduser().resolve() / "smoke").resolve()
    try:
        destination.relative_to(smoke_root)
    except ValueError as exc:
        raise ValueError(f"Smoke output must stay under {smoke_root}; got {destination}") from exc
    if destination == smoke_root or destination.suffix.casefold() != ".json":
        raise ValueError("Smoke output must be a .json file below outputs/smoke")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Smoke output already exists: {destination}; pass --overwrite to replace it"
        )
    return destination


def validate_clip(
    *,
    start_seconds: float,
    duration_seconds: float,
    video_duration_seconds: float | None,
) -> None:
    if start_seconds < 0:
        raise ValueError("--start-seconds cannot be negative")
    if not 0 < duration_seconds <= MAX_SMOKE_DURATION_SECONDS:
        raise ValueError(f"--duration-seconds must be > 0 and <= {MAX_SMOKE_DURATION_SECONDS:g}")
    if video_duration_seconds is not None and start_seconds >= video_duration_seconds:
        raise ValueError(
            f"--start-seconds ({start_seconds:g}) is outside the video "
            f"duration ({video_duration_seconds:g}s)"
        )


def extract_audio_clip(
    video_path: str | Path,
    audio_path: str | Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    ffmpeg_binary: str = "ffmpeg",
    sample_rate: int = 16_000,
    channels: int = 1,
    normalize_lufs: bool = False,
    integrated_loudness: float = -16.0,
) -> None:
    """Extract a mono 16 kHz WAV clip with no shell interpolation."""

    source = Path(video_path)
    destination = Path(audio_path)
    command = [
        ffmpeg_binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration_seconds:.6f}",
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
    ]
    if normalize_lufs:
        command.extend(
            ["-af", f"loudnorm=I={integrated_loudness}:TP=-1.5:LRA=11"]
        )
    command.extend(["-c:a", "pcm_s16le", str(destination)])
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg executable was not found: {ffmpeg_binary!r}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"Cannot extract smoke audio from {source}: {detail}")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no smoke audio: {destination}")


def build_smoke_payload(
    document: TranscriptDocument,
    *,
    source_video: Path,
    model_reference: str,
    device: str,
    compute_type: str,
    requested_language: str | None,
    start_seconds: float,
    requested_duration_seconds: float,
    extraction_seconds: float,
    transcription_seconds: float,
    total_seconds: float,
) -> dict[str, object]:
    """Wrap clip-relative ASR in a deliberately non-canonical smoke schema."""

    segments: list[dict[str, object]] = []
    for segment in document.segments:
        words: list[dict[str, object]] = []
        for word in segment.words:
            words.append(
                {
                    "text": word.text,
                    "start": word.start,
                    "end": word.end,
                    "absolute_start": start_seconds + word.start,
                    "absolute_end": start_seconds + word.end,
                    "confidence": word.confidence,
                }
            )
        segments.append(
            {
                "segment_id": segment.segment_id,
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "absolute_start": start_seconds + segment.start,
                "absolute_end": start_seconds + segment.end,
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
                "confidence": segment.confidence,
                "words": words,
            }
        )
    measured_duration = (
        document.duration_seconds
        if document.duration_seconds is not None
        else requested_duration_seconds
    )
    return {
        "schema_version": "1.0",
        "kind": "phowhisper_smoke",
        "created_at": utc_now(),
        "video_id": document.video_id,
        "source_video": str(source_video.resolve()),
        "clip": {
            "start_seconds": start_seconds,
            "requested_duration_seconds": requested_duration_seconds,
            "measured_audio_duration_seconds": measured_duration,
            "absolute_end_seconds": start_seconds + measured_duration,
            "timestamp_note": (
                "start/end are clip-relative; absolute_start/absolute_end are "
                "timestamps in the source video"
            ),
        },
        "model": {
            "reference": model_reference,
            "device": device,
            "compute_type": compute_type,
            "requested_language": requested_language,
        },
        "detected_language": document.language,
        "language_probability": document.language_probability,
        "text": document.text,
        "segments": segments,
        "timing_seconds": {
            "audio_extraction": extraction_seconds,
            "transcription_including_model_load": transcription_seconds,
            "total": total_seconds,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--video-id",
        required=True,
        help="One video from the configured scope, for example L21_V001",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Source-video start timestamp (default: 0)",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=15.0,
        help=f"Short clip duration, at most {MAX_SMOKE_DURATION_SECONDS:g}s (default: 15)",
    )
    parser.add_argument(
        "--output",
        help="JSON path below outputs/smoke (a safe default is generated)",
    )
    parser.add_argument("--model", help="Override the local PhoWhisper CT2 model")
    parser.add_argument("--device", help="Override inference device")
    parser.add_argument("--compute-type", help="Override CTranslate2 compute type")
    parser.add_argument(
        "--language",
        help="Language code; use 'auto' for detection",
    )
    parser.add_argument("--beam-size", type=int, help="Override ASR beam size")
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    adapter_factory: Callable[[ASRConfig], FasterWhisperAdapter] = FasterWhisperAdapter,
) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    configure_logging(args.verbose)

    config = load_config(project_path(args.config))
    manifest = load_configured_manifest(config)
    videos = select_videos(
        manifest,
        configured_ids=config.scope.video_ids,
        requested_ids=[args.video_id],
    )
    video = videos[0]
    source = video.resolve_path(manifest.dataset_root, "video_path")
    if source is None or not source.is_file():
        raise FileNotFoundError(f"Source video is missing for {video.video_id}: {source}")

    validate_clip(
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
        video_duration_seconds=video.duration_seconds,
    )
    raw_output = args.output or default_output_path(
        config.paths.outputs_root,
        video_id=video.video_id,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
    )
    candidate_output = project_path(raw_output) if args.output else Path(raw_output)
    output = validate_output_path(
        candidate_output,
        outputs_root=config.paths.outputs_root,
        overwrite=args.overwrite,
    )

    model_reference = _model_reference(args.model or config.asr.model)
    model_path = Path(model_reference)
    if model_path.is_absolute() and not model_path.is_dir():
        raise FileNotFoundError(
            f"Local PhoWhisper model not found: {model_path}. Run `python "
            "scripts/prepare_phowhisper.py` first."
        )
    language = (
        config.asr.language
        if args.language is None
        else (None if args.language.casefold() == "auto" else args.language)
    )
    beam_size = args.beam_size or config.asr.beam_size
    if beam_size < 1:
        raise ValueError("--beam-size must be positive")
    adapter_config = ASRConfig(
        model_name=model_reference,
        device=args.device or config.asr.device,
        compute_type=args.compute_type or config.asr.compute_type,
        language=language,
        beam_size=beam_size,
        vad_filter=config.asr.vad_filter,
        vad_min_silence_duration_ms=config.asr.vad_min_silence_duration_ms,
        word_timestamps=config.asr.word_timestamps,
        condition_on_previous_text=config.asr.condition_on_previous_text,
        hotwords=config.asr.hotwords,
    )

    total_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="agentforce-phowhisper-smoke-") as temporary:
        audio_path = Path(temporary) / f"{video.video_id}.wav"
        LOGGER.info(
            "Extracting %.3fs at %.3fs from %s",
            args.duration_seconds,
            args.start_seconds,
            source,
        )
        extraction_started = time.perf_counter()
        extract_audio_clip(
            source,
            audio_path,
            start_seconds=args.start_seconds,
            duration_seconds=args.duration_seconds,
            ffmpeg_binary=args.ffmpeg_binary,
            sample_rate=config.asr.audio_sample_rate,
            channels=config.asr.audio_channels,
            normalize_lufs=config.asr.normalize_lufs,
            integrated_loudness=config.asr.integrated_loudness,
        )
        extraction_seconds = time.perf_counter() - extraction_started

        LOGGER.info("Loading/transcribing with %s", model_reference)
        transcription_started = time.perf_counter()
        document = adapter_factory(adapter_config).transcribe(
            audio_path,
            video_id=video.video_id,
        )
        transcription_seconds = time.perf_counter() - transcription_started

    total_seconds = time.perf_counter() - total_started
    payload = build_smoke_payload(
        document,
        source_video=source,
        model_reference=model_reference,
        device=adapter_config.device,
        compute_type=adapter_config.compute_type,
        requested_language=language,
        start_seconds=args.start_seconds,
        requested_duration_seconds=args.duration_seconds,
        extraction_seconds=extraction_seconds,
        transcription_seconds=transcription_seconds,
        total_seconds=total_seconds,
    )
    atomic_write_json(output, payload)
    print(
        json.dumps(
            {
                "status": "completed",
                "video_id": video.video_id,
                "clip": payload["clip"],
                "model": payload["model"],
                "segment_count": len(document.segments),
                "text": document.text,
                "timing_seconds": payload["timing_seconds"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
