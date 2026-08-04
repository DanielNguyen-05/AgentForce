"""Run one Known-Item Search query and return exact frame predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.runtime import (
    build_dense_refiner,
    build_text_searchers,
    load_search_fields,
)
from agentforce.tasks import KISConfig, KISSolver
from agentforce.utils.io import atomic_write_json


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve and rank exact frames for a Known-Item Search query."
    )
    parser.add_argument("--query", required=True, help="Description of the target scene")
    parser.add_argument("--query-id", default="KIS001", help="Identifier written to the result")
    parser.add_argument("--limit", type=int, default=100, help="Maximum predictions (1-100)")
    parser.add_argument(
        "--dense-refine",
        action="store_true",
        help="Decode video around top coarse hits to refine the exact frame",
    )
    parser.add_argument(
        "--dense-top-n",
        type=int,
        default=20,
        help="Number of coarse candidates to refine when --dense-refine is enabled",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the TOML config")
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args(argv)


def solve_kis(
    query: str,
    *,
    query_id: str,
    limit: int,
    dense_refine: bool,
    dense_top_n: int,
    config_path: Path,
) -> dict:
    if not 1 <= limit <= 100:
        raise ValueError("--limit must be between 1 and 100")
    if dense_top_n < 0:
        raise ValueError("--dense-top-n must be non-negative")

    config = load_config(config_path)
    fields = load_search_fields(config)
    searchers = build_text_searchers(fields)
    weights = {field.name: field.weight for field in fields}
    refiner = build_dense_refiner(config) if dense_refine else None
    solver = KISSolver(
        searchers,
        dense_refiner=refiner,
        config=KISConfig(
            per_source_k=config.retrieval.top_keyframes,
            top_k=limit,
            rank_constant=config.retrieval.rrf_k,
            modality_weights=weights,
            temporal_radius_seconds=config.retrieval.duplicate_seconds,
            dense_refine_top_n=dense_top_n if refiner is not None else 0,
        ),
    )
    answers = solver.solve(query)
    return {
        "query_id": query_id,
        "task_type": "kis",
        "predictions": [
            {
                "video_id": answer.video_id,
                "frame_ids": [answer.frame_idx],
                "score": answer.score,
                "candidate_id": answer.candidate_id,
                "timestamp": answer.timestamp,
                "evidence": dict(answer.evidence),
            }
            for answer in answers
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    payload = solve_kis(
        args.query,
        query_id=args.query_id,
        limit=args.limit,
        dense_refine=args.dense_refine,
        dense_top_n=args.dense_top_n,
        config_path=project_path(args.config),
    )
    if args.output:
        atomic_write_json(project_path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
