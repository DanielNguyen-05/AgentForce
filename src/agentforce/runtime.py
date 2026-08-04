"""Runtime factories shared by the direct Python scripts.

This module is deliberately separate from command-line parsing.  A script can
load a config and call these functions directly, while notebooks and tests can
reuse exactly the same runtime assembly without invoking another entry point.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from agentforce.config import AppConfig
from agentforce.data.schemas import DatasetManifest
from agentforce.engine import IndexTextSearcher, MultimodalSearchEngine, SearchField


def load_dataset_manifest(config: AppConfig) -> DatasetManifest:
    """Load the persisted dataset manifest, or build one when it is absent."""

    manifest_path = config.paths.artifacts_root / "manifests" / "dataset.json"
    if manifest_path.exists():
        return DatasetManifest.read_json(manifest_path)

    from agentforce.data.manifest import build_manifest

    return build_manifest(config.paths.dataset_root, validate=False)


def _read_index_manifest(index_root: Path, name: str) -> tuple[dict[str, object], Path]:
    manifest_path = index_root / f"{name}.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing index manifest: {manifest_path}. Rebuild the corresponding index."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Index manifest must contain a JSON object: {manifest_path}")
    if payload.get("builder_version") != 1:
        raise ValueError(f"Unsupported or stale index manifest: {manifest_path}")
    return payload, manifest_path


def _resolve_index_paths(
    config: AppConfig,
    name: str,
) -> tuple[Path, Path]:
    """Validate an index manifest and return its vector/metadata paths."""

    index_root = config.paths.artifacts_root / "indexes"
    manifest, manifest_path = _read_index_manifest(index_root, name)

    if name.startswith("visual"):
        expected_encoder = manifest.get("expected_encoder")
        configured_encoder = {
            "model": config.embeddings.clip_model,
            "pretrained": config.embeddings.clip_pretrained,
        }
        if expected_encoder and expected_encoder != configured_encoder:
            raise ValueError(
                f"Visual query encoder in the config does not match {manifest_path}"
            )
    elif manifest.get("encoder_model") not in {None, config.embeddings.text_model}:
        raise ValueError(f"Text query encoder in the config does not match {manifest_path}")

    if name.endswith("_windows"):
        windows_manifest_path = (
            config.paths.artifacts_root / "windows" / "temporal_windows.manifest.json"
        )
        if not windows_manifest_path.is_file():
            raise FileNotFoundError(
                "Temporal-window manifest is missing. Run "
                "`python scripts/build_windows.py` and rebuild the window indexes."
            )
        windows_manifest = json.loads(windows_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("window_records_hash") != windows_manifest.get(
            "canonical_records_hash"
        ):
            raise ValueError(
                f"Index {name!r} is stale relative to temporal_windows.jsonl; rebuild it"
            )

    vectors_path = index_root / str(manifest.get("vectors_path", f"{name}.npy"))
    metadata_path = index_root / str(manifest.get("metadata_path", f"{name}.jsonl"))
    for path, kind in ((vectors_path, "vectors"), (metadata_path, "metadata")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {kind} file declared by {manifest_path}: {path}")
    return vectors_path, metadata_path


def load_search_fields(config: AppConfig) -> list[SearchField]:
    """Open all available validated indexes and their matching query encoders."""

    from agentforce.embeddings.encoders import (
        OpenCLIPTextEncoder,
        SentenceTransformerEncoder,
    )
    from agentforce.indexing.numpy_index import NumpyIndex

    index_root = config.paths.artifacts_root / "indexes"
    fields: list[SearchField] = []

    visual_window_vectors = index_root / "visual_windows.npy"
    visual_keyframe_vectors = index_root / "visual_keyframes.npy"
    visual_name: str | None = None
    if visual_window_vectors.is_file():
        visual_name = "visual_windows"
    elif visual_keyframe_vectors.is_file():
        visual_name = "visual_keyframes"

    if visual_name is not None:
        vectors_path, metadata_path = _resolve_index_paths(config, visual_name)
        fields.append(
            SearchField(
                name="visual",
                index=NumpyIndex(vectors_path, metadata_path),
                encoder=OpenCLIPTextEncoder(
                    config.embeddings.clip_model,
                    config.embeddings.clip_pretrained,
                    device=config.embeddings.device,
                ),
                top_k=(
                    config.retrieval.top_windows
                    if visual_name == "visual_windows"
                    else config.retrieval.top_keyframes
                ),
                weight=config.retrieval.weights.get("visual", 1.0),
            )
        )

    text_modalities = tuple(
        name
        for name in ("asr", "ocr", "caption", "objects", "metadata")
        if (index_root / f"{name}_windows.npy").is_file()
    )
    if text_modalities:
        encoder = SentenceTransformerEncoder(
            config.embeddings.text_model,
            device=config.embeddings.device,
        )
        for name in text_modalities:
            vectors_path, metadata_path = _resolve_index_paths(config, f"{name}_windows")
            fields.append(
                SearchField(
                    name=name,
                    index=NumpyIndex(vectors_path, metadata_path),
                    encoder=encoder,
                    top_k=config.retrieval.top_windows,
                    weight=config.retrieval.weights.get(name, 1.0),
                )
            )

    if not fields:
        raise FileNotFoundError(
            f"No search indexes found in {index_root}. Run "
            "`python scripts/build_visual_indexes.py` first."
        )
    return fields


def build_search_engine(
    config: AppConfig,
    *,
    fields: list[SearchField] | None = None,
) -> MultimodalSearchEngine:
    """Create the high-level multimodal search engine."""

    return MultimodalSearchEngine(
        fields if fields is not None else load_search_fields(config),
        rank_constant=config.retrieval.rrf_k,
        duplicate_seconds=config.retrieval.duplicate_seconds,
    )


def build_text_searchers(
    fields: list[SearchField],
) -> dict[str, IndexTextSearcher]:
    """Adapt loaded fields to the small search interface used by task solvers."""

    return {field.name: IndexTextSearcher(field) for field in fields}


def build_gemini_verifier(config: AppConfig):
    """Create the cached Gemini verifier using the API key named in config."""

    from agentforce.gemini import (
        GeminiClientConfig,
        GeminiQAClient,
        GoogleGenAITransport,
        JsonFileCache,
        QAVerifier,
    )

    api_key = os.getenv(config.gemini.api_key_env)
    if not api_key:
        raise ValueError(
            f"Environment variable {config.gemini.api_key_env} is not set. "
            "Set it before running run_qa.py."
        )
    client = GeminiQAClient(
        GeminiClientConfig(
            model=config.gemini.model,
            max_attempts=config.gemini.max_attempts,
        ),
        transport=GoogleGenAITransport(api_key=api_key),
    )
    return QAVerifier(
        client,
        cache=JsonFileCache(config.paths.artifacts_root / "gemini" / "cache"),
        audit_path=config.paths.artifacts_root / "gemini" / "calls.jsonl",
    )


def build_dense_refiner(config: AppConfig):
    """Create the optional OpenCV/OpenCLIP exact-frame refinement runtime."""

    from agentforce.temporal import (
        DenseFrameRefiner,
        OpenCLIPFrameScorer,
        OpenCVFrameSource,
    )

    manifest = load_dataset_manifest(config)
    video_paths = {
        video.video_id: video_path
        for video in manifest.videos
        if (video_path := video.resolve_path(manifest.dataset_root, "video_path")) is not None
    }
    if not video_paths:
        raise FileNotFoundError(
            f"No video files are available under the dataset root {manifest.dataset_root}"
        )
    return DenseFrameRefiner(
        OpenCVFrameSource(video_paths),
        OpenCLIPFrameScorer(
            config.embeddings.clip_model,
            config.embeddings.clip_pretrained,
            device=config.embeddings.device,
        ),
        radius_seconds=config.trake.refine_radius_seconds,
        sample_fps=config.trake.dense_fps,
    )


__all__ = [
    "build_dense_refiner",
    "build_gemini_verifier",
    "build_search_engine",
    "build_text_searchers",
    "load_dataset_manifest",
    "load_search_fields",
]
