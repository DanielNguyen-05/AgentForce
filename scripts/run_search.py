"""Search all available multimodal indexes for one text query."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Sequence

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.runtime import build_search_engine
from agentforce.utils.io import atomic_write_json


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the visual/text indexes and print ranked candidates as JSON."
    )
    parser.add_argument("--query", required=True, help="Natural-language search query")
    parser.add_argument(
        "--task",
        choices=("kis", "qa", "trake"),
        default=None,
        help="Optional task hint; otherwise it is inferred from the query",
    )
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of results")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the TOML config")
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args(argv)


def search_payload(query: str, *, task: str | None, limit: int, config_path: Path) -> dict:
    if limit < 1:
        raise ValueError("--limit must be positive")
    config = load_config(config_path)
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    payload = search_payload(
        args.query,
        task=args.task,
        limit=args.limit,
        config_path=project_path(args.config),
    )
    if args.output:
        atomic_write_json(project_path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
