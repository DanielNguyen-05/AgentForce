"""Typed TOML configuration with deterministic path resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import tomllib

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class PathConfig:
    dataset_root: Path = Path("dataset")
    artifacts_root: Path = Path("artifacts")
    outputs_root: Path = Path("outputs")

    @property
    def transcripts_dir(self) -> Path:
        """Canonical ASR artifacts, colocated with the other derived data."""

        return self.artifacts_root / "transcripts"


@dataclass(frozen=True, slots=True)
class ScopeConfig:
    """Explicit video subset used by every canonical artifact builder."""

    video_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(str(item).strip().upper() for item in self.video_ids)
        if any(not item for item in normalized):
            raise ConfigurationError("scope.video_ids must not contain empty values")
        if len(normalized) != len(set(normalized)):
            raise ConfigurationError("scope.video_ids must not contain duplicates")
        object.__setattr__(self, "video_ids", normalized)


@dataclass(frozen=True, slots=True)
class WindowConfig:
    length_seconds: float = 10.0
    stride_seconds: float = 5.0


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    top_keyframes: int = 500
    top_windows: int = 200
    top_videos: int = 30
    rrf_k: int = 60
    duplicate_seconds: float = 2.0
    visual_level: str = "keyframe"
    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "visual": 1.0,
            "asr": 0.8,
            "ocr": 0.8,
            "metadata": 0.5,
            "objects": 0.3,
        }
    )


@dataclass(frozen=True, slots=True)
class ASRConfig:
    model: str = "models/phowhisper-large-ct2"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = "vi"
    beam_size: int = 5
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 500
    word_timestamps: bool = True
    condition_on_previous_text: bool = True
    hotwords: tuple[str, ...] = ()
    audio_sample_rate: int = 16_000
    audio_channels: int = 1
    normalize_lufs: bool = True
    integrated_loudness: float = -16.0

    def __post_init__(self) -> None:
        normalized = tuple(
            value for item in self.hotwords if (value := str(item).strip())
        )
        object.__setattr__(self, "hotwords", normalized)


@dataclass(frozen=True, slots=True)
class OCRConfig:
    backend: str = "tesseract"
    languages: str = "vie+eng"
    minimum_confidence: float = 0.30


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    # OpenAI's original CLIP ViT-B/32 uses QuickGELU.  This OpenCLIP model
    # name avoids a query/image encoder mismatch warning.
    clip_model: str = "ViT-B-32-quickgelu"
    clip_pretrained: str = "openai"
    text_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    device: str = "cpu"
    batch_size: int = 128
    query_local_files_only: bool = True


@dataclass(frozen=True, slots=True)
class GeminiConfig:
    model: str = "gemini-3.6-flash"
    api_key_env: str = "GEMINI_API_KEY"
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    max_output_tokens: int = 2048
    max_retry_output_tokens: int = 8192
    thinking_level: str | None = "minimal"
    prompt_version: str = "qa-multiframe-v1"
    max_candidates: int = 12


@dataclass(frozen=True, slots=True)
class TrakeConfig:
    dense_fps: float = 12.5
    refine_radius_seconds: float = 3.0
    strict_order: bool = True
    min_event_gap_seconds: float = 0.0
    max_event_gap_seconds: float | None = None
    transition_penalty: float = 0.0


@dataclass(frozen=True, slots=True)
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    scope: ScopeConfig = field(default_factory=ScopeConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    trake: TrakeConfig = field(default_factory=TrakeConfig)
    source_path: Path | None = None


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Configuration section [{name}] must be a table")
    return value


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(path: str | Path = "configs/default.toml") -> AppConfig:
    """Load configuration with paths resolved from the owning project root."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {source}")
    with source.open("rb") as handle:
        raw = tomllib.load(handle)

    # The standard config lives in ``<project>/configs``.  A standalone config
    # resolves beside itself, which keeps temporary/test configs intuitive.
    base_dir = source.parent.parent if source.parent.name == "configs" else source.parent
    paths = _section(raw, "paths")
    scope = _section(raw, "scope")
    windows = _section(raw, "windows")
    retrieval = _section(raw, "retrieval")
    embeddings = _section(raw, "embeddings")
    asr = _section(raw, "asr")
    ocr = _section(raw, "ocr")
    gemini = _section(raw, "gemini")
    trake = _section(raw, "trake")

    config = AppConfig(
        paths=PathConfig(
            dataset_root=_resolve_path(paths.get("dataset_root", "dataset"), base_dir),
            artifacts_root=_resolve_path(paths.get("artifacts_root", "artifacts"), base_dir),
            outputs_root=_resolve_path(paths.get("outputs_root", "outputs"), base_dir),
        ),
        scope=ScopeConfig(video_ids=tuple(scope.get("video_ids", ()))),
        windows=WindowConfig(**windows),
        retrieval=RetrievalConfig(
            top_keyframes=int(retrieval.get("top_keyframes", 500)),
            top_windows=int(retrieval.get("top_windows", 200)),
            top_videos=int(retrieval.get("top_videos", 30)),
            rrf_k=int(retrieval.get("rrf_k", 60)),
            duplicate_seconds=float(retrieval.get("duplicate_seconds", 2.0)),
            visual_level=str(retrieval.get("visual_level", "keyframe")),
            weights=dict(retrieval.get("weights", {})) or RetrievalConfig().weights,
        ),
        embeddings=EmbeddingConfig(**embeddings),
        asr=ASRConfig(**asr),
        ocr=OCRConfig(**ocr),
        gemini=GeminiConfig(**gemini),
        trake=TrakeConfig(**trake),
        source_path=source,
    )
    _validate_config(config)
    return config


def _validate_config(config: AppConfig) -> None:
    if config.windows.length_seconds <= 0 or config.windows.stride_seconds <= 0:
        raise ConfigurationError("Window length and stride must be positive")
    if config.windows.stride_seconds > config.windows.length_seconds:
        raise ConfigurationError("Window stride should not exceed window length")
    if config.retrieval.visual_level not in {"keyframe", "window"}:
        raise ConfigurationError("retrieval.visual_level must be 'keyframe' or 'window'")
    if config.asr.beam_size < 1:
        raise ConfigurationError("asr.beam_size must be positive")
    if config.asr.vad_min_silence_duration_ms < 0:
        raise ConfigurationError("asr.vad_min_silence_duration_ms cannot be negative")
    if config.asr.audio_sample_rate < 1 or config.asr.audio_channels < 1:
        raise ConfigurationError("ASR audio sample rate and channels must be positive")
    if config.gemini.max_attempts < 1:
        raise ConfigurationError("gemini.max_attempts must be at least 1")
    if config.gemini.timeout_seconds <= 0:
        raise ConfigurationError("gemini.timeout_seconds must be positive")
    if config.gemini.max_output_tokens < 1:
        raise ConfigurationError("gemini.max_output_tokens must be positive")
    if config.gemini.max_retry_output_tokens < config.gemini.max_output_tokens:
        raise ConfigurationError(
            "gemini.max_retry_output_tokens must be >= gemini.max_output_tokens"
        )
    if config.gemini.thinking_level not in {None, "minimal", "low", "medium", "high"}:
        raise ConfigurationError(
            "gemini.thinking_level must be minimal, low, medium, high, or null"
        )
    if not config.gemini.prompt_version.strip():
        raise ConfigurationError("gemini.prompt_version must not be empty")
    if not 1 <= config.gemini.max_candidates <= 64:
        raise ConfigurationError("gemini.max_candidates must be between 1 and 64")
    if not 0 <= config.ocr.minimum_confidence <= 1:
        raise ConfigurationError("ocr.minimum_confidence must be between 0 and 1")
    if config.embeddings.batch_size < 1:
        raise ConfigurationError("embeddings.batch_size must be positive")
    if config.trake.dense_fps <= 0 or config.trake.refine_radius_seconds < 0:
        raise ConfigurationError("TRAKE dense settings are invalid")
    if config.trake.min_event_gap_seconds < 0 or config.trake.transition_penalty < 0:
        raise ConfigurationError("TRAKE gaps and transition penalty must be non-negative")
    if (
        config.trake.max_event_gap_seconds is not None
        and config.trake.max_event_gap_seconds < config.trake.min_event_gap_seconds
    ):
        raise ConfigurationError("trake.max_event_gap_seconds is smaller than the minimum")
