"""Evaluate ranked JSON predictions with the AIC metrics.

Run:
    python scripts/evaluate_results.py \
        --ground-truth examples/ground_truth.json \
        --predictions examples/predictions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import project_path

from agentforce.evaluation import (
    TaskType,
    evaluate_run,
    parse_ground_truth,
    parse_prediction_document,
)
from agentforce.utils.io import atomic_write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, help="Ground-truth JSON file")
    parser.add_argument("--predictions", required=True, help="Ranked predictions JSON file")
    parser.add_argument("--output", help="Optional report JSON file")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    ground_truth_path = project_path(args.ground_truth)
    predictions_path = project_path(args.predictions)
    raw_ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    raw_predictions = json.loads(predictions_path.read_text(encoding="utf-8"))

    ground_truths = {
        str(query_id): parse_ground_truth(value)
        for query_id, value in raw_ground_truth.items()
    }
    only_ground_truth_id = next(iter(ground_truths)) if len(ground_truths) == 1 else None
    predictions = parse_prediction_document(
        raw_predictions,
        query_id=only_ground_truth_id if isinstance(raw_predictions, list) else None,
    )
    if (
        isinstance(raw_predictions, dict)
        and "predictions" in raw_predictions
        and "task_type" in raw_predictions
    ):
        wrapper_id = str(raw_predictions.get("query_id", "")).strip()
        if wrapper_id in ground_truths:
            wrapper_task = TaskType.parse(raw_predictions["task_type"])
            expected_task = ground_truths[wrapper_id].task_type
            if wrapper_task is not expected_task:
                raise ValueError(
                    f"Prediction output task {wrapper_task.value!r} does not match "
                    f"ground-truth task {expected_task.value!r} for {wrapper_id}"
                )
    report = evaluate_run(ground_truths, predictions).to_dict()
    if args.output:
        atomic_write_json(project_path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
