#!/usr/bin/env python3
"""Inspect dependencies, executables, and configured dataset directories."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Sequence

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.data.layout import DatasetLayout
from agentforce.utils.io import atomic_write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="TOML config path (default: configs/default.toml)",
    )
    parser.add_argument("--output", help="Optional JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    config = load_config(project_path(args.config))
    layout = DatasetLayout.from_path(config.paths.dataset_root)

    modules = {
        "numpy": "core",
        "cv2": "video",
        "faiss": "faiss",
        "faster_whisper": "asr",
        "google.genai": "gemini",
        "open_clip": "vision",
        "PIL": "ocr/vision",
        "pytesseract": "ocr",
        "sentence_transformers": "semantic",
    }
    dependencies: dict[str, dict[str, object]] = {}
    for module, extra in modules.items():
        try:
            installed = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError):
            installed = False
        dependencies[module] = {"installed": installed, "extra": extra}

    report = {
        "python": sys.version,
        "config": str(config.source_path),
        "dataset_root": str(layout.root),
        "dataset_components": {
            name: {"exists": path.exists(), "path": str(path)}
            for name, path in layout.component_roots().items()
        },
        "executables": {
            name: shutil.which(name) for name in ("ffmpeg", "ffprobe", "tesseract")
        },
        "dependencies": dependencies,
        "gemini_key_configured": bool(os.getenv(config.gemini.api_key_env)),
    }
    if args.output:
        atomic_write_json(project_path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
