"""Lazy Faster-Whisper adapter producing canonical timestamped transcripts."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentforce.data.schemas import (
    TranscriptDocument,
    TranscriptSegment,
    TranscriptWord,
)

from .audio import AudioExtractionConfig, extract_audio


class ASRDependencyError(RuntimeError):
    pass


class TranscriptSchemaError(ValueError):
    """Raised when a transcript is neither valid canonical nor legacy PhoWhisper JSON."""


@dataclass(frozen=True, slots=True)
class ASRConfig:
    model_name: str = "large-v3"
    device: str = "auto"
    compute_type: str = "default"
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 500
    word_timestamps: bool = True
    condition_on_previous_text: bool = True
    hotwords: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if self.vad_min_silence_duration_ms < 0:
            raise ValueError("vad_min_silence_duration_ms cannot be negative")
        normalized = tuple(
            value for item in self.hotwords if (value := str(item).strip())
        )
        object.__setattr__(self, "hotwords", normalized)


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
            vad_parameters=(
                {"min_silence_duration_ms": self.config.vad_min_silence_duration_ms}
                if self.config.vad_filter
                else None
            ),
            word_timestamps=self.config.word_timestamps,
            condition_on_previous_text=self.config.condition_on_previous_text,
            hotwords=", ".join(self.config.hotwords) or None,
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


def read_transcript(
    path: str | Path,
    *,
    rewrite_legacy: bool = False,
) -> TranscriptDocument:
    """Read canonical or legacy PhoWhisper JSON with strict field validation.

    The three initial PhoWhisper trial files used an ``audio-first`` schema with
    ``time_range`` and ``audio_metadata`` nested inside each segment.  That
    layout is accepted explicitly and converted to the canonical records used
    by window construction.  Set ``rewrite_legacy`` to atomically replace a
    legacy source file with its canonical representation after it validates.
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TranscriptSchemaError(
            f"Invalid transcript JSON at {source}: {exc.msg} (line {exc.lineno})"
        ) from exc
    document, schema = _parse_transcript_payload(payload)
    if rewrite_legacy and schema == "legacy_phowhisper":
        write_transcript(document, source)
    return document


def migrate_transcript(
    source_path: str | Path,
    destination_path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> TranscriptDocument:
    """Validate a transcript and atomically write it in canonical form.

    When ``destination_path`` is omitted, a legacy file is migrated in place.
    Supplying a distinct destination is useful when preserving the original
    trial output.  An existing distinct destination is never replaced unless
    ``overwrite`` is explicitly enabled.
    """

    source = Path(source_path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TranscriptSchemaError(
            f"Invalid transcript JSON at {source}: {exc.msg} (line {exc.lineno})"
        ) from exc
    document, _ = _parse_transcript_payload(payload)
    destination = source if destination_path is None else Path(destination_path)
    same_destination = destination.expanduser().resolve() == source.expanduser().resolve()
    if not same_destination and destination.exists() and not overwrite:
        raise FileExistsError(
            f"Transcript destination already exists: {destination}; pass overwrite=True"
        )
    write_transcript(document, destination)
    return document


def _parse_transcript_payload(payload: object) -> tuple[TranscriptDocument, str]:
    root = _mapping(payload, "transcript")
    canonical = "video_id" in root or "model_name" in root
    legacy = "source_video" in root or "pipeline_config" in root or "audio_info" in root
    if canonical and legacy:
        raise TranscriptSchemaError(
            "Transcript mixes canonical and legacy PhoWhisper root fields"
        )
    if canonical:
        return _parse_canonical_transcript(root), "canonical"
    if legacy:
        return _parse_legacy_phowhisper_transcript(root), "legacy_phowhisper"
    raise TranscriptSchemaError(
        "Unsupported transcript schema: expected canonical 'video_id' or legacy "
        "PhoWhisper 'source_video' fields"
    )


def _parse_canonical_transcript(root: Mapping[str, object]) -> TranscriptDocument:
    video_id = _string(_required(root, "video_id", "transcript"), "transcript.video_id")
    model_name = _string(
        _required(root, "model_name", "transcript"), "transcript.model_name"
    )
    language = _nullable_string(
        _required(root, "language", "transcript"), "transcript.language"
    )
    language_probability = _nullable_probability(
        _required(root, "language_probability", "transcript"),
        "transcript.language_probability",
    )
    duration = _nullable_nonnegative_number(
        _required(root, "duration_seconds", "transcript"),
        "transcript.duration_seconds",
    )
    source_path = _nullable_string(root.get("source_path"), "transcript.source_path")
    segments = _parse_segments(
        _sequence(_required(root, "segments", "transcript"), "transcript.segments"),
        legacy=False,
    )
    _validate_document_timing(segments, duration)
    kwargs: dict[str, object] = {
        "video_id": video_id,
        "language": language,
        "language_probability": language_probability,
        "duration_seconds": duration,
        "segments": segments,
        "model_name": model_name,
        "source_path": source_path,
    }
    if "created_at" in root:
        kwargs["created_at"] = _string(root["created_at"], "transcript.created_at")
    return TranscriptDocument(**kwargs)  # type: ignore[arg-type]


def _parse_legacy_phowhisper_transcript(
    root: Mapping[str, object],
) -> TranscriptDocument:
    source_video = _string(
        _required(root, "source_video", "transcript"), "transcript.source_video"
    )
    # A legacy file may have been generated on Windows and later read on macOS/Linux.
    video_id = Path(source_video.replace("\\", "/")).stem
    if not video_id:
        raise TranscriptSchemaError("transcript.source_video must identify a video")
    pipeline = _mapping(
        _required(root, "pipeline_config", "transcript"), "transcript.pipeline_config"
    )
    audio_info = _mapping(
        _required(root, "audio_info", "transcript"), "transcript.audio_info"
    )
    model_name = _string(
        _required(pipeline, "asr_model", "transcript.pipeline_config"),
        "transcript.pipeline_config.asr_model",
    )
    language = _nullable_string(
        _required(audio_info, "detected_language", "transcript.audio_info"),
        "transcript.audio_info.detected_language",
    )
    language_probability = _nullable_probability(
        _required(audio_info, "language_probability", "transcript.audio_info"),
        "transcript.audio_info.language_probability",
    )
    duration = _nullable_nonnegative_number(
        _required(audio_info, "duration_sec", "transcript.audio_info"),
        "transcript.audio_info.duration_sec",
    )
    segments = _parse_segments(
        _sequence(_required(root, "segments", "transcript"), "transcript.segments"),
        legacy=True,
    )
    _validate_document_timing(segments, duration)
    kwargs: dict[str, object] = {
        "video_id": video_id,
        "language": language,
        "language_probability": language_probability,
        "duration_seconds": duration,
        "segments": segments,
        "model_name": model_name,
        "source_path": source_video,
    }
    if "generated_at" in root:
        kwargs["created_at"] = _string(root["generated_at"], "transcript.generated_at")
    return TranscriptDocument(**kwargs)  # type: ignore[arg-type]


def _parse_segments(
    items: Sequence[object],
    *,
    legacy: bool,
) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    seen_ids: set[int] = set()
    previous_start = -math.inf
    for index, value in enumerate(items):
        location = f"transcript.segments[{index}]"
        item = _mapping(value, location)
        segment_id = _nonnegative_integer(
            _required(item, "segment_id", location), f"{location}.segment_id"
        )
        if segment_id in seen_ids:
            raise TranscriptSchemaError(f"Duplicate transcript segment_id: {segment_id}")
        seen_ids.add(segment_id)
        if legacy:
            timing = _mapping(
                _required(item, "time_range", location), f"{location}.time_range"
            )
            metadata = _mapping(
                _required(item, "audio_metadata", location),
                f"{location}.audio_metadata",
            )
            start = _nonnegative_number(
                _required(timing, "start", f"{location}.time_range"),
                f"{location}.time_range.start",
            )
            end = _nonnegative_number(
                _required(timing, "end", f"{location}.time_range"),
                f"{location}.time_range.end",
            )
            clean_text = _string(
                _required(metadata, "clean_text", f"{location}.audio_metadata"),
                f"{location}.audio_metadata.clean_text",
                allow_empty=True,
            )
            raw_text = _string(
                _required(metadata, "raw_text", f"{location}.audio_metadata"),
                f"{location}.audio_metadata.raw_text",
                allow_empty=True,
            )
            text = clean_text.strip() or raw_text.strip()
            words = _parse_words(metadata.get("words", []), location=location, legacy=True)
            avg_logprob = None
            no_speech_prob = _nullable_probability(
                metadata.get("no_speech_prob"), f"{location}.audio_metadata.no_speech_prob"
            )
            confidence = _nullable_probability(
                metadata.get("confidence"), f"{location}.audio_metadata.confidence"
            )
        else:
            start = _nonnegative_number(
                _required(item, "start", location), f"{location}.start"
            )
            end = _nonnegative_number(_required(item, "end", location), f"{location}.end")
            text = _string(
                _required(item, "text", location), f"{location}.text", allow_empty=True
            )
            words = _parse_words(item.get("words", []), location=location, legacy=False)
            avg_logprob = _nullable_number(item.get("avg_logprob"), f"{location}.avg_logprob")
            no_speech_prob = _nullable_probability(
                item.get("no_speech_prob"), f"{location}.no_speech_prob"
            )
            confidence = _nullable_probability(
                item.get("confidence"), f"{location}.confidence"
            )
        if end < start:
            raise TranscriptSchemaError(f"{location}.end must be >= start")
        if start < previous_start:
            raise TranscriptSchemaError("Transcript segments must be ordered by start time")
        previous_start = start
        _validate_word_timing(words, start, end, location)
        segments.append(
            TranscriptSegment(
                segment_id=segment_id,
                start=start,
                end=end,
                text=text,
                words=words,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                confidence=confidence,
            )
        )
    return segments


def _parse_words(
    value: object,
    *,
    location: str,
    legacy: bool,
) -> list[TranscriptWord]:
    words: list[TranscriptWord] = []
    for index, raw_word in enumerate(_sequence(value, f"{location}.words")):
        word_location = f"{location}.words[{index}]"
        item = _mapping(raw_word, word_location)
        text_field = "word" if legacy else "text"
        confidence_field = "probability" if legacy else "confidence"
        words.append(
            TranscriptWord(
                text=_string(
                    _required(item, text_field, word_location),
                    f"{word_location}.{text_field}",
                    # Faster-Whisper can occasionally emit a timestamped empty
                    # token after whitespace stripping. Preserve it for a lossless
                    # round trip; window alignment ignores unusable empty tokens.
                    allow_empty=True,
                ),
                start=_nonnegative_number(
                    _required(item, "start", word_location), f"{word_location}.start"
                ),
                end=_nonnegative_number(
                    _required(item, "end", word_location), f"{word_location}.end"
                ),
                confidence=_nullable_probability(
                    item.get(confidence_field), f"{word_location}.{confidence_field}"
                ),
            )
        )
    return words


def _validate_word_timing(
    words: Sequence[TranscriptWord],
    segment_start: float,
    segment_end: float,
    location: str,
) -> None:
    previous_start = -math.inf
    tolerance = 0.05
    for index, word in enumerate(words):
        if word.end < word.start:
            raise TranscriptSchemaError(f"{location}.words[{index}].end must be >= start")
        if word.start < previous_start:
            raise TranscriptSchemaError(f"{location}.words must be ordered by start time")
        if word.start < segment_start - tolerance or word.end > segment_end + tolerance:
            raise TranscriptSchemaError(
                f"{location}.words[{index}] falls outside its segment time range"
            )
        previous_start = word.start


def _validate_document_timing(
    segments: Sequence[TranscriptSegment], duration: float | None
) -> None:
    if duration is None:
        return
    for segment in segments:
        if segment.end > duration + 0.05:
            raise TranscriptSchemaError(
                f"Segment {segment.segment_id} ends after transcript duration ({duration})"
            )


def _required(mapping: Mapping[str, object], key: str, location: str) -> object:
    if key not in mapping:
        raise TranscriptSchemaError(f"Missing required field: {location}.{key}")
    return mapping[key]


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TranscriptSchemaError(f"{location} must be a JSON object")
    return value


def _sequence(value: object, location: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TranscriptSchemaError(f"{location} must be a JSON array")
    return value


def _string(value: object, location: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TranscriptSchemaError(f"{location} must be a string")
    if not allow_empty and not value.strip():
        raise TranscriptSchemaError(f"{location} cannot be empty")
    return value


def _nullable_string(value: object, location: str) -> str | None:
    if value is None:
        return None
    return _string(value, location)


def _integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TranscriptSchemaError(f"{location} must be an integer")
    return value


def _nonnegative_integer(value: object, location: str) -> int:
    result = _integer(value, location)
    if result < 0:
        raise TranscriptSchemaError(f"{location} must be non-negative")
    return result


def _number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TranscriptSchemaError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise TranscriptSchemaError(f"{location} must be finite")
    return result


def _nullable_number(value: object, location: str) -> float | None:
    return None if value is None else _number(value, location)


def _nonnegative_number(value: object, location: str) -> float:
    result = _number(value, location)
    if result < 0:
        raise TranscriptSchemaError(f"{location} must be non-negative")
    return result


def _nullable_nonnegative_number(value: object, location: str) -> float | None:
    return None if value is None else _nonnegative_number(value, location)


def _nullable_probability(value: object, location: str) -> float | None:
    if value is None:
        return None
    result = _number(value, location)
    if not 0 <= result <= 1:
        raise TranscriptSchemaError(f"{location} must be between 0 and 1")
    return result


def _attribute(value: Any, name: str, default: Any) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None
