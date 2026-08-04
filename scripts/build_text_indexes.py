#!/usr/bin/env python3
"""Build separate ASR, OCR, object, and metadata text indexes."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.data.manifest import build_manifest
from agentforce.data.schemas import DatasetManifest
from agentforce.data.scope import (
    assert_exact_scope,
    normalize_video_ids,
    scope_hash,
    select_videos,
)
from agentforce.embeddings.encoders import HashingTextEncoder, SentenceTransformerEncoder
from agentforce.indexing.build_windows import build_window_text_indexes, read_window_records
from agentforce.utils.io import atomic_write_json

# Video metadata is identical for every window of one video. Repeating that
# vector across hundreds of windows creates arbitrary tie ranks and can boost
# unrelated frames, so it remains opt-in until a proper video-level prior is
# implemented.
DEFAULT_FIELDS = "asr_text,ocr_text,object_labels"
ALLOWED_FIELDS = {
    "asr_text",
    "ocr_text",
    "object_labels",
    "metadata_text",
}


def _manifest(config) -> DatasetManifest:
    path = config.paths.artifacts_root / "manifests" / "dataset.json"
    return DatasetManifest.read_json(path) if path.exists() else build_manifest(
        config.paths.dataset_root, validate=False
    )


def _parse_fields(raw: str) -> tuple[str, ...]:
    fields = tuple(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    if not fields:
        raise ValueError("--fields must contain at least one field")
    unsupported = sorted(set(fields) - ALLOWED_FIELDS)
    if unsupported:
        raise ValueError(f"Unsupported fields: {', '.join(unsupported)}")
    return fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--windows", help="Temporal-window JSONL path")
    parser.add_argument("--output-dir", help="Index output directory")
    parser.add_argument("--fields", default=DEFAULT_FIELDS)
    parser.add_argument("--batch-size", type=int, help="Override embeddings.batch_size")
    parser.add_argument("--model", help="Override the sentence-transformer model")
    parser.add_argument("--device", help="Override the embedding device")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help=(
            "Allow Hugging Face network access when the semantic model is not cached; "
            "without this flag the configured local-files-only policy is respected"
        ),
    )
    parser.add_argument(
        "--encoder",
        choices=("sentence-transformer", "hashing"),
        default="sentence-transformer",
        help="Use hashing only for dependency-light smoke tests",
    )
    parser.add_argument("--hash-dimension", type=int, default=512)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--video-ids",
        "--video-id",
        dest="video_ids",
        nargs="+",
        metavar="VIDEO_ID",
        help="Build an explicit subset inside the configured scope",
    )
    selection.add_argument(
        "--all", action="store_true", help="Build the complete configured scope"
    )
    parser.add_argument("--limit", type=int, help="Limit windows for a smoke run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.hash_dimension < 1:
        raise ValueError("--hash-dimension must be positive")
    config = load_config(project_path(args.config))
    manifest = _manifest(config)
    videos = select_videos(
        manifest,
        configured_ids=config.scope.video_ids,
        requested_ids=args.video_ids,
        all_configured=args.all,
    )
    selected_video_ids = tuple(sorted(video.video_id for video in videos))
    batch_size = args.batch_size or config.embeddings.batch_size
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    fields = _parse_fields(args.fields)
    windows_path = (
        project_path(args.windows)
        if args.windows
        else config.paths.artifacts_root / "windows" / "temporal_windows.jsonl"
    )
    canonical_output_dir = config.paths.artifacts_root / "indexes"
    output_dir = project_path(args.output_dir) if args.output_dir else canonical_output_dir
    canonical_build = output_dir.resolve() == canonical_output_dir.resolve()
    if canonical_build:
        if args.limit is not None:
            raise ValueError("--limit requires an explicit non-canonical --output-dir")
        expected_ids = config.scope.video_ids or tuple(
            video.video_id for video in manifest.videos
        )
        assert_exact_scope(
            selected_video_ids,
            expected_ids,
            label="Canonical text indexes",
        )

    rows = read_window_records(windows_path)
    input_video_ids = normalize_video_ids(str(row.get("video_id", "")) for row in rows)
    if canonical_build:
        assert_exact_scope(
            input_video_ids,
            selected_video_ids,
            label="Temporal windows used by canonical text indexes",
        )
    selected_set = set(selected_video_ids)
    rows = [
        row
        for row in rows
        if str(row.get("video_id", "")).strip().upper() in selected_set
    ]
    present_ids = set(normalize_video_ids(str(row.get("video_id", "")) for row in rows))
    missing_ids = sorted(selected_set - present_ids)
    if missing_ids:
        raise ValueError(
            "No temporal windows found for selected video IDs: " + ", ".join(missing_ids)
        )
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No temporal windows selected")

    if "metadata_text" in fields:
        metadata_by_video = {
            video.video_id: "\n".join(
                part.strip()
                for part in (video.title or "", video.description or "", video.author or "")
                if part.strip()
            )
            for video in videos
        }
        for row in rows:
            row["metadata_text"] = metadata_by_video.get(str(row["video_id"]), "")

    if args.encoder == "hashing":
        encoder = HashingTextEncoder(dimension=args.hash_dimension)
    else:
        encoder = SentenceTransformerEncoder(
            args.model or config.embeddings.text_model,
            device=args.device or config.embeddings.device,
            local_files_only=(
                config.embeddings.query_local_files_only
                and not args.allow_model_download
            ),
        )
    manifests = build_window_text_indexes(
        rows,
        output_dir,
        encoder,
        fields=fields,
        batch_size=batch_size,
    )
    indexed_video_ids = sorted(
        normalize_video_ids(str(row.get("video_id", "")) for row in rows)
    )
    expected_encoder = {
        "class": type(encoder).__name__,
        "model": getattr(encoder, "model_name", None),
        "dimension": int(encoder.dimension),
    }
    for modality, index_manifest in manifests.items():
        index_manifest.update(
            {
                "video_ids": indexed_video_ids,
                "scope_hash": scope_hash(indexed_video_ids),
                "expected_encoder": expected_encoder,
            }
        )
        atomic_write_json(
            output_dir / f"{modality}_windows.manifest.json", index_manifest
        )
    print(json.dumps(manifests, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
