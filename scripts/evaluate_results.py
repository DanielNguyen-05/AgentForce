"""Evaluate ranked JSON predictions with the AIC metrics.

Run:
    python scripts/evaluate_results.py \
        --ground-truth examples/ground_truth.json \
        --predictions examples/predictions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from _bootstrap import project_path

from agentforce.evaluation import evaluate_run, parse_ground_truth, parse_prediction
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
    predictions = {
        str(query_id): tuple(parse_prediction(value) for value in values)
        for query_id, values in raw_predictions.items()
    }
    report = evaluate_run(ground_truths, predictions).to_dict()
    if args.output:
        atomic_write_json(project_path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
