#!/usr/bin/env python3
"""Create a standalone HTML gallery from local retrieval or an existing result JSON."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence
import webbrowser

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import AppConfig, load_config
from agentforce.runtime import build_search_engine, load_dataset_manifest
from agentforce.utils.io import atomic_write_json
from agentforce.visualization import (
    KeyframeRegistry,
    normalize_result_payload,
    render_gallery_html,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize ranked keyframes as one self-contained HTML file. Give --input "
            "to inspect existing Search/KIS/QA/TRAKE JSON, or give --query to run local "
            "retrieval first. Query mode never calls Gemini."
        )
    )
    parser.add_argument(
        "--input",
        "--results",
        dest="input_path",
        help="Existing run_search/KIS/QA/TRAKE JSON",
    )
    parser.add_argument(
        "--query",
        help=(
            "Run this local retrieval query when --input is absent; with --input, "
            "use it only as the displayed query label for older result files"
        ),
    )
    parser.add_argument(
        "--task",
        choices=("kis", "qa", "trake"),
        help="Optional local retrieval task hint (query mode only)",
    )
    parser.add_argument(
        "--max-results",
        "--limit",
        dest="max_results",
        type=int,
        default=20,
        help="Maximum ranked results to retrieve/render (default: 20)",
    )
    parser.add_argument("--columns", type=int, default=3, help="Desktop gallery columns")
    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=640,
        help="Maximum embedded thumbnail width in pixels",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=82,
        help="Embedded JPEG quality from 30 to 95",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to TOML config")
    parser.add_argument(
        "--output",
        help="HTML output; defaults to artifacts/visualizations/<result-or-query>.html",
    )
    parser.add_argument(
        "--json-output",
        help="Raw retrieval JSON path in query mode (default: next to HTML)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML in the default browser",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.input_path is None and not (args.query and args.query.strip()):
        raise ValueError("Provide --input RESULT.json or --query TEXT")
    if args.input_path is not None and args.task is not None:
        raise ValueError("--task is only valid when running a new --query")
    if not 1 <= args.max_results <= 100:
        raise ValueError("--max-results must be between 1 and 100")


def _search_payload(
    query: str,
    *,
    task: str | None,
    limit: int,
    config: AppConfig,
) -> dict[str, object]:
    result = build_search_engine(config).search(query, task_type=task, limit=limit)
    return {
        "query": {
            "raw_text": result.query.raw_text,
            "task_type": result.query.task_type.value,
            "retrieval_text": result.query.retrieval_text,
            "question": result.query.question,
            "language": result.query.language,
            "expected_answer_type": result.query.expected_answer_type,
            "events": [asdict(event) for event in result.query.events],
        },
        "variants": [asdict(variant) for variant in result.variants],
        "source_counts": dict(result.source_counts),
        "hits": [
            {
                "rank": rank,
                "candidate_id": hit.candidate_id,
                "score": hit.score,
                "modality_scores": dict(hit.modality_scores),
                "metadata": dict(hit.metadata),
            }
            for rank, hit in enumerate(result.hits, 1)
        ],
    }


def _load_payload(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid result JSON at {path}: {exc.msg} (line {exc.lineno})"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"Result JSON must contain one object: {path}")
    return value


def _default_output(
    config: AppConfig,
    *,
    input_path: Path | None,
    query: str | None,
) -> Path:
    if input_path is not None:
        name = input_path.stem
    else:
        assert query is not None
        digest = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:10]
        name = f"query_{digest}"
    return config.paths.artifacts_root / "visualizations" / f"{name}.html"


def _atomic_write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    _validate_args(args)
    config = load_config(project_path(args.config))
    input_path = project_path(args.input_path) if args.input_path else None

    generated_json: Path | None = None
    if input_path is not None:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        payload = _load_payload(input_path)
        source_label = str(input_path)
        query_override = args.query
    else:
        assert args.query is not None
        payload = _search_payload(
            args.query,
            task=args.task,
            limit=args.max_results,
            config=config,
        )
        source_label = "local retrieval generated by visualize_results.py"
        query_override = None

    output = (
        project_path(args.output)
        if args.output
        else _default_output(config, input_path=input_path, query=args.query)
    )
    if output.suffix.casefold() != ".html":
        raise ValueError("--output must use the .html extension")

    if input_path is None:
        generated_json = (
            project_path(args.json_output) if args.json_output else output.with_suffix(".json")
        )
        atomic_write_json(generated_json, payload)
    elif args.json_output:
        raise ValueError("--json-output is only valid in query mode")

    registry = KeyframeRegistry(load_dataset_manifest(config))
    document = normalize_result_payload(
        payload,
        registry,
        source_label=source_label,
        query_override=query_override,
        max_results=args.max_results,
    )
    html = render_gallery_html(
        document,
        columns=args.columns,
        thumbnail_width=args.thumbnail_width,
        jpeg_quality=args.jpeg_quality,
    )
    _atomic_write_text(output, html)

    summary = {
        "status": "completed",
        "task_type": document.task_type,
        "query": document.query_text,
        "ranked_results": len({record.prediction_rank for record in document.records}),
        "frames": len(document.records),
        "html": str(output),
        "retrieval_json": str(generated_json) if generated_json is not None else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.open:
        webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
