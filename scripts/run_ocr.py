#!/usr/bin/env python3
"""Run OCR directly on keyframes from selected videos."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.data.manifest import build_manifest
from agentforce.data.schemas import DatasetManifest
from agentforce.data.scope import assert_exact_scope, select_videos
from agentforce.data.timeline import timeline_from_manifest
from agentforce.logging_utils import configure_logging
from agentforce.preprocessing.ocr import EasyOCREngine, OCRProcessor, TesseractOCREngine

LOGGER = logging.getLogger("scripts.run_ocr")


def _manifest(config) -> DatasetManifest:
    path = config.paths.artifacts_root / "manifests" / "dataset.json"
    return DatasetManifest.read_json(path) if path.exists() else build_manifest(
        config.paths.dataset_root, validate=False
    )


def _easyocr_languages(raw: str) -> tuple[str, ...]:
    aliases = {"vie": "vi", "eng": "en"}
    values = [item for item in re.split(r"[+,]", raw) if item]
    return tuple(aliases.get(item.casefold(), item.casefold()) for item in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--video-ids",
        "--video-id",
        dest="video_ids",
        nargs="+",
        metavar="VIDEO_ID",
        help="OCR an explicit subset inside the configured scope",
    )
    selection.add_argument(
        "--all", action="store_true", help="OCR the complete configured scope"
    )
    parser.add_argument("--limit", type=int, help="Limit selected videos for a smoke run")
    parser.add_argument("--frame-limit", type=int, help="Limit keyframes per video")
    parser.add_argument("--output-dir", help="OCR JSONL directory")
    parser.add_argument("--backend", choices=("tesseract", "easyocr"))
    parser.add_argument("--languages", help="OCR languages, e.g. vie+eng or vi,en")
    parser.add_argument("--minimum-confidence", type=float)
    parser.add_argument("--tesseract-psm", type=int, default=11)
    parser.add_argument("--gpu", action="store_true", help="Use GPU with EasyOCR")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "EasyOCR keyframes per detector batch (default: 1; batched images must "
            "have equal dimensions)"
        ),
    )
    parser.add_argument(
        "--show-backend-warnings",
        action="store_true",
        help="Show repetitive low-level EasyOCR/PyTorch warnings for debugging",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    if args.frame_limit is not None and args.frame_limit < 1:
        raise ValueError("--frame-limit must be positive")
    if args.tesseract_psm < 0:
        raise ValueError("--tesseract-psm cannot be negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    configure_logging(args.verbose)
    config = load_config(project_path(args.config))
    manifest = _manifest(config)
    videos = select_videos(
        manifest,
        configured_ids=config.scope.video_ids,
        requested_ids=args.video_ids,
        all_configured=args.all,
        limit=args.limit,
    )
    canonical_output_dir = config.paths.artifacts_root / "ocr"
    output_dir = project_path(args.output_dir) if args.output_dir else canonical_output_dir
    if output_dir.resolve() == canonical_output_dir.resolve():
        if args.limit is not None:
            raise ValueError("--limit requires an explicit non-canonical --output-dir")
        if args.frame_limit is not None:
            raise ValueError("--frame-limit requires an explicit non-canonical --output-dir")
        expected_ids = config.scope.video_ids or tuple(
            video.video_id for video in manifest.videos
        )
        assert_exact_scope(
            (video.video_id for video in videos),
            expected_ids,
            label="Canonical OCR artifacts",
        )
    backend = args.backend or config.ocr.backend
    languages = args.languages or config.ocr.languages
    minimum_confidence = (
        config.ocr.minimum_confidence
        if args.minimum_confidence is None
        else args.minimum_confidence
    )
    if backend == "tesseract":
        engine = TesseractOCREngine(
            language=languages, config=f"--psm {args.tesseract_psm}"
        )
    else:
        engine = EasyOCREngine(
            languages=_easyocr_languages(languages),
            gpu=args.gpu,
            batch_size=args.batch_size,
            suppress_pin_memory_warnings=not args.show_backend_warnings,
        )
    processor = OCRProcessor(engine, min_confidence=minimum_confidence)

    completed = skipped = 0
    for video in videos:
        destination = output_dir / f"{video.video_id}.jsonl"
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue
        timeline = timeline_from_manifest(manifest, video.video_id)
        if args.frame_limit is not None:
            timeline = timeline[: args.frame_limit]
        LOGGER.info("OCR %s (%d keyframes)", video.video_id, len(timeline))
        processor.process_many(
            (
                (record.keyframe_uid, record.image_path)
                for record in timeline
                if record.image_path is not None
            ),
            destination,
        )
        completed += 1

    print(
        json.dumps(
            {"completed": completed, "skipped": skipped, "output_dir": str(output_dir)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
