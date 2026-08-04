"""Safe ffmpeg helpers used by ASR and dense-video preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any


class FFmpegError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AudioExtractionConfig:
    sample_rate: int = 16_000
    channels: int = 1
    normalize_lufs: bool = False
    integrated_loudness: float = -16.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.channels <= 0:
            raise ValueError("sample_rate and channels must be positive")


@dataclass(frozen=True, slots=True)
class MediaProbe:
    duration_seconds: float | None
    fps: float | None
    frame_count: int | None
    width: int | None
    height: int | None
    has_audio: bool


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
    *,
    config: AudioExtractionConfig | None = None,
    overwrite: bool = False,
    ffmpeg_binary: str = "ffmpeg",
) -> Path:
    """Extract mono PCM WAV atomically; no shell expansion is involved."""

    source = Path(video_path)
    output = Path(output_path)
    settings = config or AudioExtractionConfig()
    if output.suffix.lower() != ".wav":
        raise ValueError("extract_audio output_path must use the .wav extension")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not overwrite:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".partial.wav", dir=output.parent
    )
    import os

    os.close(descriptor)
    temporary = Path(temporary_name)
    command = [
        ffmpeg_binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        str(settings.channels),
        "-ar",
        str(settings.sample_rate),
    ]
    if settings.normalize_lufs:
        command.extend(
            [
                "-af",
                f"loudnorm=I={settings.integrated_loudness}:TP=-1.5:LRA=11",
            ]
        )
    command.extend(["-c:a", "pcm_s16le", str(temporary)])
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown ffmpeg failure"
            raise FFmpegError(f"Audio extraction failed for {source}: {detail}")
        temporary.replace(output)
    except FileNotFoundError as exc:
        raise FFmpegError(f"ffmpeg executable not found: {ffmpeg_binary}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return output


def probe_media(video_path: str | Path, *, ffprobe_binary: str = "ffprobe") -> MediaProbe:
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(source),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise FFmpegError(f"ffprobe executable not found: {ffprobe_binary}") from exc
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or f"Could not probe {source}")
    try:
        payload: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned invalid JSON for {source}") from exc
    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    duration = _float_or_none(video.get("duration")) or _float_or_none(
        payload.get("format", {}).get("duration")
    )
    fps = _parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    frame_count = _int_or_none(video.get("nb_frames"))
    if frame_count is None and duration is not None and fps is not None:
        frame_count = round(duration * fps)
    return MediaProbe(
        duration_seconds=duration,
        fps=fps,
        frame_count=frame_count,
        width=_int_or_none(video.get("width")),
        height=_int_or_none(video.get("height")),
        has_audio=has_audio,
    )


def _parse_rate(value: object) -> float | None:
    if value is None:
        return None
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else None
    return _float_or_none(text)


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
