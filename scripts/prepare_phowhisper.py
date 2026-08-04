#!/usr/bin/env python3
"""Download/convert VinAI PhoWhisper to a validated local CTranslate2 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from _bootstrap import project_path

from agentforce.preprocessing.phowhisper_model import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_MODEL,
    QUANTIZATION_TYPES,
    ConversionRequest,
    PhoWhisperConversionError,
    convert_phowhisper,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--quantization", choices=QUANTIZATION_TYPES, default="int8")
    parser.add_argument("--revision", help="Optional Hugging Face revision/commit")
    parser.add_argument(
        "--low-cpu-mem-usage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ask Transformers to reduce peak CPU memory (requires Accelerate)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Safely replace an existing converted model after validating the new one",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    request = ConversionRequest(
        source_model=args.source_model,
        output_dir=project_path(args.output_dir),
        quantization=args.quantization,
        revision=args.revision,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        force=args.force,
    )
    try:
        result = convert_phowhisper(request)
    except (PhoWhisperConversionError, OSError, RuntimeError, ValueError) as exc:
        print(f"PhoWhisper preparation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
