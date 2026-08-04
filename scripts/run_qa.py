"""Retrieve local frames, then rank grounded Q&A predictions with Gemini."""

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
        description=(
            "Retrieve/rank frames locally, then ask Gemini only about those selected "
            "frames. Ranked batches are the competition default."
        )
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
        "--single-answer",
        action="store_true",
        help=(
            "Cheap smoke-test mode: make one Gemini call and retain at most one answer "
            "instead of competition-style ranked batches"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Selected local frames per Gemini call in ranked mode (default: 4)",
    )
    parser.add_argument(
        "--answer-limit",
        type=int,
        default=100,
        help="Maximum ranked predictions retained (default: 100; ignored with --single-answer)",
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
    single_answer: bool = False,
) -> dict:
    config = load_config(config_path)
    candidate_limit = (
        config.gemini.max_candidates if max_candidates is None else max_candidates
    )
    if not 1 <= candidate_limit <= 64:
        raise ValueError("--max-candidates must be between 1 and 64")
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if answer_limit < 1:
        raise ValueError("--answer-limit must be positive")

    combined_query = query.strip()
    if question is not None:
        if not question.strip():
            raise ValueError("--question must not be empty")
        combined_query = f"{combined_query}\n{question.strip()}"

    # Preflight the key/configuration before loading heavyweight retrieval
    # models. Constructing the verifier does not make an API request; Gemini is
    # invoked only later by QASolver, after local retrieval has selected frames.
    verifier = build_gemini_verifier(config)
    fields = load_search_fields(config)
    search_result = build_search_engine(config, fields=fields).search(
        combined_query,
        task_type="qa",
        limit=candidate_limit * 3,
    )
    parsed = search_result.query
    if not parsed.question:
        raise ValueError("The Q&A question could not be parsed from the supplied text")

    solver = QASolver(
        verifier,
        load_dataset_manifest(config),
        config=QAConfig(
            max_candidates=candidate_limit,
            language=parsed.language,
            batch_size=batch_size,
        ),
    )
    final_result = None
    if single_answer:
        # Cheap smoke-test path: local retrieval/candidate resolution still happens
        # first, and the one Gemini call receives only that selected local set.
        final_result = solver.solve(
            query_id=query_id,
            question=parsed.question,
            retrieval_context=parsed.retrieval_text,
            candidates=search_result.hits,
        )
        predictions = (
            [
                {
                    "video_id": final_result.supporting_video_id,
                    "frame_ids": [final_result.supporting_frame_idx],
                    "answer": final_result.normalized_answer,
                    "score": final_result.confidence,
                    "gemini": final_result.to_dict(),
                }
            ]
            if (
                final_result.answerable
                and final_result.supporting_video_id is not None
                and final_result.supporting_frame_idx is not None
                and final_result.normalized_answer is not None
            )
            else []
        )
    else:
        # Competition path: each paid call sees one disjoint batch drawn only
        # from frames already retrieved, ranked, and resolved on this machine.
        answers = solver.solve_ranked(
            query_id=query_id,
            question=parsed.question,
            retrieval_context=parsed.retrieval_text,
            candidates=search_result.hits,
        )[:answer_limit]
        predictions = [
            {
                "video_id": answer.result.supporting_video_id,
                "frame_ids": [answer.result.supporting_frame_idx],
                "answer": answer.result.normalized_answer,
                "score": answer.score,
                "gemini": answer.result.to_dict(),
            }
            for answer in answers
        ]
    return {
        "query_id": query_id,
        "task_type": "qa",
        "qa_mode": "single_final" if single_answer else "ranked_batches",
        "question": parsed.question,
        "retrieval_context": parsed.retrieval_text,
        "retrieved_candidates": len(search_result.hits),
        "predictions": predictions,
        "final_gemini": final_result.to_dict() if final_result is not None else None,
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
        single_answer=args.single_answer,
    )
    if args.output:
        atomic_write_json(project_path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
