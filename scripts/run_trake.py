"""Run one ordered multi-event TRAKE query."""

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
from agentforce.tasks import TRAKEConfig, TRAKESolver
from agentforce.utils.io import atomic_write_json


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find videos containing a supplied sequence of ordered events."
    )
    parser.add_argument(
        "--query",
        required=True,
        help=(
            "At least two ordered events separated by 'sau đó', 'rồi', '->', "
            "or numbered lines"
        ),
    )
    parser.add_argument("--query-id", default="TRAKE001", help="Identifier in the result")
    parser.add_argument("--limit", type=int, default=100, help="Maximum predictions (1-100)")
    parser.add_argument(
        "--dense-refine",
        action="store_true",
        help="Decode video near coarse hits before enforcing exact event order",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the TOML config")
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args(argv)


def solve_trake(
    query: str,
    *,
    query_id: str,
    limit: int,
    dense_refine: bool,
    config_path: Path,
) -> dict:
    if not 1 <= limit <= 100:
        raise ValueError("--limit must be between 1 and 100")

    config = load_config(config_path)
    fields = load_search_fields(config)
    searchers = build_text_searchers(fields)
    weights = {field.name: field.weight for field in fields}
    refiner = build_dense_refiner(config) if dense_refine else None
    solver = TRAKESolver(
        searchers,
        dense_refiner=refiner,
        config=TRAKEConfig(
            per_source_k=config.retrieval.top_keyframes,
            top_videos=config.retrieval.top_videos,
            top_k=limit,
            rank_constant=config.retrieval.rrf_k,
            modality_weights=weights,
            strict_order=config.trake.strict_order,
            min_event_gap_seconds=config.trake.min_event_gap_seconds,
            max_event_gap_seconds=config.trake.max_event_gap_seconds,
            transition_penalty=config.trake.transition_penalty,
        ),
    )
    answers = solver.solve(query)
    return {
        "query_id": query_id,
        "task_type": "trake",
        "predictions": [
            {
                "video_id": answer.video_id,
                "frame_ids": list(answer.frame_ids),
                "score": answer.score,
                "event_timestamps": list(answer.event_timestamps),
                "event_candidate_ids": list(answer.event_candidate_ids),
                "evidence": dict(answer.evidence),
            }
            for answer in answers
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    payload = solve_trake(
        args.query,
        query_id=args.query_id,
        limit=args.limit,
        dense_refine=args.dense_refine,
        config_path=project_path(args.config),
    )
    if args.output:
        atomic_write_json(project_path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
