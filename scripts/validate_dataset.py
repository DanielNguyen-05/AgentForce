#!/usr/bin/env python3
"""Build and validate the canonical dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.data.manifest import build_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="TOML config path (default: configs/default.toml)",
    )
    parser.add_argument("--dataset-root", help="Override paths.dataset_root from the config")
    parser.add_argument("--output", help="Manifest JSON path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 if validation finds an error",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    config = load_config(project_path(args.config))
    dataset_root = (
        project_path(args.dataset_root) if args.dataset_root else config.paths.dataset_root
    )
    output = (
        project_path(args.output)
        if args.output
        else config.paths.artifacts_root / "manifests" / "dataset.json"
    )

    manifest = build_manifest(dataset_root, validate=True)
    manifest.write_json(output)
    summary = {
        "manifest": str(output),
        "dataset_root": str(dataset_root),
        "videos": len(manifest.videos),
        "errors": manifest.error_count,
        "warnings": manifest.warning_count,
        "missing_video_ids": [
            video.video_id for video in manifest.videos if "video" in video.missing_components
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if args.strict and manifest.error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
