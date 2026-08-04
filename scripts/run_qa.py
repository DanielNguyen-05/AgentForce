"""Retrieve local frames, then answer one visual question with Gemini."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.runtime import (
    build_gemini_verifier,
    build_search_engine,
    load_dataset_manifest,
    load_search_fields,
)
from agentforce.tasks import QAConfig, QASolver
from agentforce.utils.io import atomic_write_json


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve candidate frames locally and ask Gemini to answer the question."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Scene/retrieval description, or a complete context-plus-question query",
    )
    parser.add_argument(
        "--question",
        help="Question to answer; recommended when --query is only the scene description",
    )
    parser.add_argument("--query-id", default="QA001", help="Identifier written to the result")
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Maximum candidate frames sent to Gemini (default: value in config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Candidate frames per independent Gemini call",
    )
    parser.add_argument(
        "--answer-limit",
        type=int,
        default=10,
        help="Maximum alternative answers retained",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the TOML config")
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args(argv)


def solve_qa(
    query: str,
    *,
    question: str | None,
    query_id: str,
    max_candidates: int | None,
    batch_size: int,
    answer_limit: int,
    config_path: Path,
) -> dict:
    config = load_config(config_path)
    candidate_limit = (
        config.gemini.max_candidates if max_candidates is None else max_candidates
    )
    if not 1 <= candidate_limit <= 64:
        raise ValueError("--max-candidates must be between 1 and 64")
    if not 1 <= batch_size <= candidate_limit:
        raise ValueError("--batch-size must be between 1 and --max-candidates")
    if answer_limit < 1:
        raise ValueError("--answer-limit must be positive")

    combined_query = query.strip()
    if question is not None:
        if not question.strip():
            raise ValueError("--question must not be empty")
        combined_query = f"{combined_query}\n{question.strip()}"

    fields = load_search_fields(config)
    search_result = build_search_engine(config, fields=fields).search(
        combined_query,
        task_type="qa",
        limit=candidate_limit * 3,
    )
    parsed = search_result.query
    if not parsed.question:
        raise ValueError("The Q&A question could not be parsed from the supplied text")

    verifier = build_gemini_verifier(config)
    solver = QASolver(
        verifier,
        load_dataset_manifest(config),
        config=QAConfig(
            max_candidates=candidate_limit,
            language=parsed.language,
            batch_size=batch_size,
        ),
    )
    answers = solver.solve_ranked(
        query_id=query_id,
        question=parsed.question,
        retrieval_context=parsed.retrieval_text,
        candidates=search_result.hits,
    )[:answer_limit]
    return {
        "query_id": query_id,
        "task_type": "qa",
        "question": parsed.question,
        "retrieval_context": parsed.retrieval_text,
        "retrieved_candidates": len(search_result.hits),
        "predictions": [
            {
                "video_id": answer.result.supporting_video_id,
                "frame_ids": [answer.result.supporting_frame_idx],
                "answer": answer.result.normalized_answer,
                "score": answer.score,
                "gemini": answer.result.to_dict(),
            }
            for answer in answers
        ],
        "gemini_diagnostics": dict(verifier.last_diagnostics),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    payload = solve_qa(
        args.query,
        question=args.question,
        query_id=args.query_id,
        max_candidates=args.max_candidates,
        batch_size=args.batch_size,
        answer_limit=args.answer_limit,
        config_path=project_path(args.config),
    )
    if args.output:
        atomic_write_json(project_path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
