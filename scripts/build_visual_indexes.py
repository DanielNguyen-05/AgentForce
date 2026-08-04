#!/usr/bin/env python3
"""Build keyframe and/or temporal-window visual indexes from supplied CLIP features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.indexing.build_visual import build_visual_index
from agentforce.indexing.build_windows import build_visual_window_index, read_window_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--level", choices=("keyframe", "window", "both"), default="both"
    )
    parser.add_argument("--windows", help="Temporal-window JSONL path")
    parser.add_argument("--output-dir", help="Index output directory")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--video-id",
        help="Window-index smoke run: keep windows from one video (not valid for keyframe level)",
    )
    parser.add_argument(
        "--limit", type=int, help="Window-index smoke run: limit the number of windows"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if (args.video_id or args.limit) and args.level in {"keyframe", "both"}:
        raise ValueError("--video-id/--limit only apply to --level window")

    config = load_config(project_path(args.config))
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else config.paths.artifacts_root / "indexes"
    )
    manifests: dict[str, object] = {}
    if args.level in {"keyframe", "both"}:
        manifests["keyframe"] = build_visual_index(
            config.paths.dataset_root, output_dir, dtype=args.dtype
        )

    if args.level in {"window", "both"}:
        windows_path = (
            project_path(args.windows)
            if args.windows
            else config.paths.artifacts_root / "windows" / "temporal_windows.jsonl"
        )
        windows = read_window_records(windows_path)
        if args.video_id:
            requested = args.video_id.upper()
            windows = [row for row in windows if str(row.get("video_id", "")).upper() == requested]
            if not windows:
                raise ValueError(f"No temporal windows found for video ID: {requested}")
        if args.limit is not None:
            windows = windows[: args.limit]
        manifests["window"] = build_visual_window_index(
            windows, config.paths.dataset_root, output_dir, dtype=args.dtype
        )

    print(json.dumps(manifests, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
