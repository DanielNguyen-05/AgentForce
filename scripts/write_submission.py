"""Write a validated task-specific submission CSV from ranked JSON.

Run:
    python scripts/write_submission.py --task kis \
        --predictions examples/predictions.json --query-id KIS001 \
        --output artifacts/submissions/KIS001.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import project_path

from agentforce.evaluation import TaskType, parse_prediction_document
from agentforce.submission import write_submission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("kis", "qa", "trake"))
    parser.add_argument("--predictions", required=True, help="Predictions JSON file")
    parser.add_argument("--query-id", help="Key to select when the JSON is a query mapping")
    parser.add_argument("--output", required=True, help="Destination CSV file")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--header", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    if not 1 <= args.limit <= 100:
        raise ValueError("--limit must be between 1 and 100")
    raw = json.loads(project_path(args.predictions).read_text(encoding="utf-8"))
    parsed_task = TaskType.parse(args.task)
    if isinstance(raw, dict) and "predictions" in raw and "task_type" in raw:
        embedded_task = TaskType.parse(raw["task_type"])
        if embedded_task is not parsed_task:
            raise ValueError(
                f"Requested task {parsed_task.value!r} does not match prediction output "
                f"task {embedded_task.value!r}"
            )
    runs = parse_prediction_document(raw, query_id=args.query_id)
    if args.query_id:
        selected_query_id = args.query_id.strip()
        if selected_query_id not in runs:
            raise ValueError(f"Query ID not found in predictions: {selected_query_id}")
    elif len(runs) == 1:
        selected_query_id = next(iter(runs))
    else:
        raise ValueError("--query-id is required when predictions contain multiple queries")
    predictions = runs[selected_query_id][: args.limit]
    output = write_submission(
        project_path(args.output),
        parsed_task,
        predictions,
        include_header=args.header,
        max_results=args.limit,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
