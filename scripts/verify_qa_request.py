"""Send a prepared local multi-frame Q&A request to Gemini."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from _bootstrap import DEFAULT_CONFIG, project_path

from agentforce.config import load_config
from agentforce.gemini import FrameCandidate, QAVerificationRequest
from agentforce.runtime import build_gemini_verifier
from agentforce.utils.io import atomic_write_json


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a prepared JSON request against local frames with Gemini."
    )
    parser.add_argument(
        "--request",
        default="examples/qa_request.json",
        help="JSON file containing question, context, and frame candidates",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the TOML config")
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args(argv)


def _load_request(path: Path, *, max_candidates: int) -> QAVerificationRequest:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Q&A request must be a JSON object: {path}")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("Q&A request field 'candidates' must be an array")

    candidates: list[FrameCandidate] = []
    for number, raw in enumerate(raw_candidates[:max_candidates], 1):
        if not isinstance(raw, dict):
            raise ValueError(f"Candidate #{number} must be a JSON object")
        item = dict(raw)
        image_path = item.get("image_path")
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError(f"Candidate #{number} must have a non-empty image_path")
        item["image_path"] = project_path(image_path)
        candidates.append(FrameCandidate(**item))
    return QAVerificationRequest(
        query_id=str(payload["query_id"]),
        question=str(payload["question"]),
        retrieval_context=str(payload.get("retrieval_context", "")),
        language=str(payload.get("language", "vi")),
        candidates=tuple(candidates),
    )


def verify_request(request_path: Path, *, config_path: Path) -> dict:
    config = load_config(config_path)
    request = _load_request(request_path, max_candidates=config.gemini.max_candidates)
    verifier = build_gemini_verifier(config)
    result = verifier.verify(request)
    payload = result.to_dict()
    payload["diagnostics"] = dict(verifier.last_diagnostics)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    payload = verify_request(
        project_path(args.request),
        config_path=project_path(args.config),
    )
    if args.output:
        atomic_write_json(project_path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
