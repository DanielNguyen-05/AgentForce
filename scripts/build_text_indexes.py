#!/usr/bin/env python3
"""Build separate ASR, OCR, caption, object, and metadata text indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.data.manifest import build_manifest
from agentforce.data.schemas import DatasetManifest
from agentforce.embeddings.encoders import HashingTextEncoder, SentenceTransformerEncoder
from agentforce.indexing.build_windows import build_window_text_indexes, read_window_records


DEFAULT_FIELDS = "asr_text,ocr_text,caption_text,object_labels,metadata_text"
ALLOWED_FIELDS = {
    "asr_text",
    "ocr_text",
    "caption_text",
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
        "--encoder",
        choices=("sentence-transformer", "hashing"),
        default="sentence-transformer",
        help="Use hashing only for dependency-light smoke tests",
    )
    parser.add_argument("--hash-dimension", type=int, default=512)
    parser.add_argument("--video-id", help="Keep windows from one video for a smoke run")
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
    batch_size = args.batch_size or config.embeddings.batch_size
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    fields = _parse_fields(args.fields)
    windows_path = (
        project_path(args.windows)
        if args.windows
        else config.paths.artifacts_root / "windows" / "temporal_windows.jsonl"
    )
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else config.paths.artifacts_root / "indexes"
    )
    rows = read_window_records(windows_path)
    if args.video_id:
        requested = args.video_id.upper()
        rows = [row for row in rows if str(row.get("video_id", "")).upper() == requested]
        if not rows:
            raise ValueError(f"No temporal windows found for video ID: {requested}")
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No temporal windows selected")

    if "metadata_text" in fields:
        manifest = _manifest(config)
        metadata_by_video = {
            video.video_id: "\n".join(
                part.strip()
                for part in (video.title or "", video.description or "", video.author or "")
                if part.strip()
            )
            for video in manifest.videos
        }
        for row in rows:
            row["metadata_text"] = metadata_by_video.get(str(row["video_id"]), "")

    if args.encoder == "hashing":
        encoder = HashingTextEncoder(dimension=args.hash_dimension)
    else:
        encoder = SentenceTransformerEncoder(
            args.model or config.embeddings.text_model,
            device=args.device or config.embeddings.device,
        )
    manifests = build_window_text_indexes(
        rows,
        output_dir,
        encoder,
        fields=fields,
        batch_size=batch_size,
    )
    print(json.dumps(manifests, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
