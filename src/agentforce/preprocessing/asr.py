"""Lazy Faster-Whisper adapter producing canonical timestamped transcripts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile
from typing import Any

from agentforce.data.schemas import (
    TranscriptDocument,
    TranscriptSegment,
    TranscriptWord,
)

from .audio import AudioExtractionConfig, extract_audio


class ASRDependencyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ASRConfig:
    model_name: str = "large-v3"
    device: str = "auto"
    compute_type: str = "default"
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = True
    condition_on_previous_text: bool = True

    def __post_init__(self) -> None:
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive")


class FasterWhisperAdapter:
    """Dependency-optional adapter; a fake model can be injected in tests."""

    def __init__(self, config: ASRConfig | None = None, *, model: Any | None = None) -> None:
        self.config = config or ASRConfig()
        self._injected_model = model
        self._loaded_model: Any | None = None

    def _model(self) -> Any:
        if self._injected_model is not None:
            return self._injected_model
        if self._loaded_model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ASRDependencyError(
                    "Faster-Whisper is optional. Install the project ASR extra "
                    "before transcription."
                ) from exc
            self._loaded_model = WhisperModel(
                self.config.model_name,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
        return self._loaded_model

    def transcribe(
        self, audio_path: str | Path, *, video_id: str | None = None
    ) -> TranscriptDocument:
        source = Path(audio_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        raw_segments, info = self._model().transcribe(
            str(source),
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            word_timestamps=self.config.word_timestamps,
            condition_on_previous_text=self.config.condition_on_previous_text,
        )
        segments: list[TranscriptSegment] = []
        for fallback_id, raw in enumerate(raw_segments):
            words = [
                TranscriptWord(
                    text=str(_attribute(word, "word", "")).strip(),
                    start=float(_attribute(word, "start", 0.0)),
                    end=float(_attribute(word, "end", 0.0)),
                    confidence=_optional_float(_attribute(word, "probability", None)),
                )
                for word in (_attribute(raw, "words", None) or [])
            ]
            avg_logprob = _optional_float(_attribute(raw, "avg_logprob", None))
            confidence = None if avg_logprob is None else min(1.0, max(0.0, math.exp(avg_logprob)))
            segments.append(
                TranscriptSegment(
                    segment_id=int(_attribute(raw, "id", fallback_id)),
                    start=float(_attribute(raw, "start", 0.0)),
                    end=float(_attribute(raw, "end", 0.0)),
                    text=str(_attribute(raw, "text", "")).strip(),
                    words=words,
                    avg_logprob=avg_logprob,
                    no_speech_prob=_optional_float(_attribute(raw, "no_speech_prob", None)),
                    confidence=confidence,
                )
            )
        return TranscriptDocument(
            video_id=video_id or source.stem,
            language=_optional_string(_attribute(info, "language", None)),
            language_probability=_optional_float(
                _attribute(info, "language_probability", None)
            ),
            duration_seconds=_optional_float(_attribute(info, "duration", None)),
            segments=segments,
            model_name=self.config.model_name,
            source_path=str(source),
        )

    def transcribe_video(
        self,
        video_path: str | Path,
        *,
        video_id: str | None = None,
        audio_config: AudioExtractionConfig | None = None,
        ffmpeg_binary: str = "ffmpeg",
    ) -> TranscriptDocument:
        source = Path(video_path)
        with tempfile.TemporaryDirectory(prefix="agentforce-asr-") as temporary_dir:
            audio_path = Path(temporary_dir) / f"{source.stem}.wav"
            extract_audio(
                source,
                audio_path,
                config=audio_config,
                ffmpeg_binary=ffmpeg_binary,
            )
            document = self.transcribe(audio_path, video_id=video_id or source.stem)
            document.source_path = str(source)
            return document


def write_transcript(document: TranscriptDocument, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(document.to_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def read_transcript(path: str | Path) -> TranscriptDocument:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    segments = []
    for item in payload.get("segments", []):
        words = [TranscriptWord(**word) for word in item.get("words", [])]
        segment_data = dict(item)
        segment_data["words"] = words
        segments.append(TranscriptSegment(**segment_data))
    data = dict(payload)
    data["segments"] = segments
    return TranscriptDocument(**data)


def _attribute(value: Any, name: str, default: Any) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None
