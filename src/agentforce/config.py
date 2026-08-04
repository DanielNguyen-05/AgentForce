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
    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "visual": 1.0,
            "asr": 0.8,
            "ocr": 0.8,
            "caption": 0.9,
            "metadata": 0.5,
            "objects": 0.3,
        }
    )


@dataclass(frozen=True, slots=True)
class ASRConfig:
    model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = "vi"
    word_timestamps: bool = True


@dataclass(frozen=True, slots=True)
class OCRConfig:
    backend: str = "tesseract"
    languages: str = "vie+eng"
    minimum_confidence: float = 0.30


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    text_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    device: str = "cpu"
    batch_size: int = 128


@dataclass(frozen=True, slots=True)
class GeminiConfig:
    model: str = "gemini-2.5-flash"
    api_key_env: str = "GEMINI_API_KEY"
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    prompt_version: str = "qa-v1"
    max_candidates: int = 12


@dataclass(frozen=True, slots=True)
class TrakeConfig:
    dense_fps: float = 12.5
    refine_radius_seconds: float = 3.0
    transition_penalty: float = 0.05


@dataclass(frozen=True, slots=True)
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
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
    """Load configuration; relative data paths resolve from the project cwd."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {source}")
    with source.open("rb") as handle:
        raw = tomllib.load(handle)

    # Dataset paths intentionally resolve from cwd so a config can be reused from
    # another checkout; users may always provide an absolute path.
    cwd = Path.cwd()
    paths = _section(raw, "paths")
    windows = _section(raw, "windows")
    retrieval = _section(raw, "retrieval")
    embeddings = _section(raw, "embeddings")
    asr = _section(raw, "asr")
    ocr = _section(raw, "ocr")
    gemini = _section(raw, "gemini")
    trake = _section(raw, "trake")

    config = AppConfig(
        paths=PathConfig(
            dataset_root=_resolve_path(paths.get("dataset_root", "dataset"), cwd),
            artifacts_root=_resolve_path(paths.get("artifacts_root", "artifacts"), cwd),
        ),
        windows=WindowConfig(**windows),
        retrieval=RetrievalConfig(
            top_keyframes=int(retrieval.get("top_keyframes", 500)),
            top_windows=int(retrieval.get("top_windows", 200)),
            top_videos=int(retrieval.get("top_videos", 30)),
            rrf_k=int(retrieval.get("rrf_k", 60)),
            duplicate_seconds=float(retrieval.get("duplicate_seconds", 2.0)),
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
    if config.gemini.max_attempts < 1:
        raise ConfigurationError("gemini.max_attempts must be at least 1")
    if not 0 <= config.ocr.minimum_confidence <= 1:
        raise ConfigurationError("ocr.minimum_confidence must be between 0 and 1")
    if config.embeddings.batch_size < 1:
        raise ConfigurationError("embeddings.batch_size must be positive")
