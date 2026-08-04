"""Write a validated task-specific submission CSV from ranked JSON.

Run:
    python scripts/write_submission.py --task kis \
        --predictions examples/predictions.json --query-id KIS001 \
        --output artifacts/submissions/KIS001.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from _bootstrap import project_path

from agentforce.evaluation import parse_prediction
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
    if isinstance(raw, dict):
        if not args.query_id:
            raise ValueError("--query-id is required when predictions JSON is a mapping")
        values = raw[args.query_id]
    else:
        values = raw
    predictions = tuple(parse_prediction(value) for value in values[: args.limit])
    output = write_submission(
        project_path(args.output),
        args.task,
        predictions,
        include_header=args.header,
        max_results=args.limit,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
