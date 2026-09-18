#!/usr/bin/env python3
"""Validate the scoped canonical artifacts without modifying them."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.data.artifact_validation import validate_artifacts
from agentforce.utils.io import atomic_write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="TOML config path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Require production visual keyframe/window and ASR/object text indexes; "
            "return exit code 2 when any error is found"
        ),
    )
    parser.add_argument(
        "--require-ocr",
        action="store_true",
        help="Treat missing, partial, or invalid OCR artifacts as errors instead of warnings",
    )
    parser.add_argument("--output", help="Optional JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    config = load_config(project_path(args.config))
    report = validate_artifacts(
        config,
        strict=args.strict,
        require_ocr=args.require_ocr,
    )
    payload = report.to_dict()
    if args.output:
        atomic_write_json(project_path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 2 if args.strict and report.error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
