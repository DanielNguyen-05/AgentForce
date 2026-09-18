"""Run every ``*-kis.txt`` query in a batch and write official KIS CSV files.

The script keeps rich JSON results outside the submission directory.  The
submission directory itself contains only headerless ``video_id,frame_id`` CSV
files named ``query-X-kis.csv``, where ``X`` is the numeric query position from
an input such as ``query-p2-X-kis.txt``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

from _bootstrap import project_path

from agentforce.config import load_config
from agentforce.evaluation import RankedPrediction
from agentforce.runtime import build_dense_refiner, build_text_searchers, load_search_fields
from agentforce.submission import write_submission
from agentforce.tasks import KISAnswer, KISConfig, KISSolver
from agentforce.utils.io import atomic_write_json, file_sha256, stable_hash


DEFAULT_INPUT_DIR = "query_batch1"
DEFAULT_OUTPUT_DIR = "outputs/query_batch1/kis"
DEFAULT_SUBMISSION_DIR = "artifacts/submissions/batch1/submission"
DEFAULT_BATCH_CONFIG = "configs/batch1_full.toml"
DEFAULT_RETRIEVAL_TEXTS = "configs/query_batch1_kis_en.json"
DEFAULT_REVIEWED_CANDIDATES = "configs/query_batch1_kis_reviewed.json"


_QUERY_STEM = re.compile(
    r"^query(?:-p(?P<batch>\d+))?-(?P<number>\d+)-kis$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ReviewedCandidate:
    """A manually verified frame that should precede automatic retrieval results."""

    video_id: str
    frame_id: int
    note: str = ""


def _natural_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    """Sort numeric filename components numerically (1, 2, 10 instead of 1, 10, 2)."""

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    )


def discover_kis_queries(input_dir: Path) -> list[Path]:
    """Return naturally sorted KIS query files, excluding QA and TRAKE files."""

    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"KIS query directory does not exist: {root}")
    paths = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.name.casefold().endswith("-kis.txt")
        ),
        key=_natural_key,
    )
    if not paths:
        raise ValueError(f"No *-kis.txt query files found in {root}")
    return paths


def submission_stem(query_path: Path) -> str:
    """Convert an official batch query stem to the requested submission stem."""

    match = _QUERY_STEM.fullmatch(Path(query_path).stem)
    if match is None:
        raise ValueError(
            "KIS query filename must match query-pN-X-kis.txt or query-X-kis.txt: "
            f"{Path(query_path).name}"
        )
    return f"query-{int(match.group('number'))}-kis"


def validate_unique_submission_stems(query_paths: Sequence[Path]) -> dict[str, str]:
    """Return source-to-submission stems and reject numeric-name collisions."""

    names: dict[str, str] = {}
    owners: dict[str, str] = {}
    for query_path in query_paths:
        source_stem = query_path.stem
        target_stem = submission_stem(query_path)
        previous = owners.get(target_stem)
        if previous is not None:
            raise ValueError(
                "KIS query filenames resolve to the same submission filename: "
                f"{previous}.txt and {query_path.name} -> {target_stem}.csv"
            )
        owners[target_stem] = source_stem
        names[source_stem] = target_stem
    return names


def read_query(path: Path) -> str:
    """Read one UTF-8 query while preserving its original wording."""

    try:
        query = Path(path).read_text(encoding="utf-8-sig").strip()
    except UnicodeError as exc:
        raise ValueError(f"Query file must be UTF-8: {path}") from exc
    if not query:
        raise ValueError(f"Query file is empty: {path}")
    return query


def load_retrieval_texts(
    path: Path | None,
    query_paths: Sequence[Path],
) -> dict[str, str]:
    """Load optional per-query retrieval text while keeping source queries intact."""

    if path is None:
        return {}
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Retrieval-text mapping does not exist: {source}") from None
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Retrieval-text mapping must be valid UTF-8 JSON: {source}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("Retrieval-text mapping must be a JSON object")

    expected = {query_path.stem for query_path in query_paths}
    missing = sorted(expected.difference(raw), key=str.casefold)
    if missing:
        raise ValueError("Retrieval-text mapping is missing query IDs: " + ", ".join(missing))
    unknown = sorted(set(raw).difference(expected), key=str.casefold)
    if unknown:
        raise ValueError("Retrieval-text mapping contains unknown query IDs: " + ", ".join(unknown))

    retrieval_texts: dict[str, str] = {}
    for query_id in expected:
        text = raw[query_id]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Retrieval text for {query_id!r} must be a non-empty string")
        retrieval_texts[query_id] = " ".join(text.split())
    return retrieval_texts


def load_reviewed_candidates(
    path: Path | None,
    query_paths: Sequence[Path],
) -> dict[str, tuple[ReviewedCandidate, ...]]:
    """Load optional human-reviewed frames without changing source query files."""

    if path is None:
        return {}
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Reviewed-candidate mapping does not exist: {source}") from None
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Reviewed-candidate mapping must be valid UTF-8 JSON: {source}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("Reviewed-candidate mapping must be a JSON object")

    expected = {query_path.stem for query_path in query_paths}
    unknown = sorted(set(raw).difference(expected), key=str.casefold)
    if unknown:
        raise ValueError(
            "Reviewed-candidate mapping contains unknown query IDs: " + ", ".join(unknown)
        )

    reviewed: dict[str, tuple[ReviewedCandidate, ...]] = {}
    for query_id, value in raw.items():
        if not isinstance(value, list) or not value:
            raise ValueError(f"Reviewed candidates for {query_id!r} must be a non-empty JSON array")
        candidates: list[ReviewedCandidate] = []
        seen: set[tuple[str, int]] = set()
        for position, item in enumerate(value, 1):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"Reviewed candidate {position} for {query_id!r} must be an object"
                )
            video_id = item.get("video_id")
            frame_id = item.get("frame_id")
            note = item.get("note", "")
            if not isinstance(video_id, str) or not re.fullmatch(r"L\d+_V\d+", video_id):
                raise ValueError(
                    f"Invalid video_id in reviewed candidate {position} for {query_id!r}"
                )
            if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
                raise ValueError(
                    f"Invalid frame_id in reviewed candidate {position} for {query_id!r}"
                )
            if not isinstance(note, str):
                raise ValueError(f"Invalid note in reviewed candidate {position} for {query_id!r}")
            key = (video_id, frame_id)
            if key in seen:
                raise ValueError(f"Duplicate reviewed candidate for {query_id!r}: {key}")
            seen.add(key)
            candidates.append(
                ReviewedCandidate(video_id=video_id, frame_id=frame_id, note=note.strip())
            )
        reviewed[query_id] = tuple(candidates)
    return reviewed


def validate_reviewed_candidates_against_dataset(
    reviewed: Mapping[str, Sequence[ReviewedCandidate]],
    dataset_manifest_path: Path,
) -> None:
    """Ensure every reviewed frame addresses an existing in-bounds video frame."""

    source = Path(dataset_manifest_path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset manifest does not exist: {source}") from None
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Dataset manifest must be valid UTF-8 JSON: {source}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("videos"), list):
        raise ValueError(f"Dataset manifest has no videos array: {source}")

    frame_counts: dict[str, int] = {}
    for position, item in enumerate(payload["videos"], 1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Invalid video record {position} in dataset manifest: {source}")
        video_id = item.get("video_id")
        frame_count = item.get("frame_count")
        if not isinstance(video_id, str):
            raise ValueError(f"Missing video_id in record {position} of dataset manifest: {source}")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
            raise ValueError(f"Invalid frame_count for {video_id!r} in dataset manifest: {source}")
        frame_counts[video_id] = frame_count

    for query_id, candidates in reviewed.items():
        for candidate in candidates:
            frame_count = frame_counts.get(candidate.video_id)
            if frame_count is None:
                raise ValueError(
                    f"Reviewed candidate for {query_id!r} references unknown video "
                    f"{candidate.video_id!r}"
                )
            if candidate.frame_id >= frame_count:
                raise ValueError(
                    f"Reviewed frame for {query_id!r} is outside {candidate.video_id}: "
                    f"frame_id={candidate.frame_id}, valid range=0..{frame_count - 1}"
                )


def _remove_stale_kis_files(root: Path, expected_stems: set[str], suffix: str) -> list[str]:
    """Remove generated KIS files whose query no longer exists in this batch."""

    removed: list[str] = []
    for path in sorted(root.glob(f"*-kis{suffix}"), key=_natural_key):
        if path.stem not in expected_stems:
            path.unlink()
            removed.append(str(path.resolve()))
    return removed


def _file_fingerprint(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    return {"path": str(source.resolve()), "sha256": file_sha256(source)}


def apply_reviewed_candidates(
    answers: Sequence[object],
    reviewed: Sequence[ReviewedCandidate],
    *,
    limit: int,
) -> tuple[object, ...]:
    """Prepend verified frames, then fill remaining slots with retrieval results."""

    if not reviewed:
        return tuple(answers[:limit])

    by_key = {(answer.video_id, answer.frame_idx): answer for answer in answers}
    top_score = max((float(answer.score) for answer in answers), default=0.0)
    ranked: list[object] = []
    seen: set[tuple[str, int]] = set()
    for position, candidate in enumerate(reviewed):
        key = (candidate.video_id, candidate.frame_id)
        existing = by_key.get(key)
        evidence = dict(existing.evidence) if existing is not None else {}
        evidence["human_reviewed"] = True
        if candidate.note:
            evidence["review_note"] = candidate.note
        ranked.append(
            KISAnswer(
                video_id=candidate.video_id,
                frame_idx=candidate.frame_id,
                score=top_score + (len(reviewed) - position) * 1e-6,
                candidate_id=(
                    existing.candidate_id
                    if existing is not None
                    else f"{candidate.video_id}_F{candidate.frame_id:09d}_REVIEWED"
                ),
                timestamp=existing.timestamp if existing is not None else None,
                evidence=evidence,
            )
        )
        seen.add(key)

    for answer in answers:
        key = (answer.video_id, answer.frame_idx)
        if key not in seen:
            ranked.append(answer)
            seen.add(key)
        if len(ranked) >= limit:
            break
    return tuple(ranked[:limit])


def _result_payload(
    query_id: str,
    query: str,
    retrieval_text: str,
    answers: Sequence[object],
) -> dict:
    predictions: list[dict] = []
    for answer in answers:
        predictions.append(
            {
                "video_id": answer.video_id,
                "frame_ids": [answer.frame_idx],
                "score": answer.score,
                "candidate_id": answer.candidate_id,
                "timestamp": answer.timestamp,
                "evidence": dict(answer.evidence),
            }
        )
    return {
        "query": query,
        "query_id": query_id,
        "retrieval_text": retrieval_text,
        "task_type": "kis",
        "predictions": predictions,
    }


def run_batch(
    input_dir: Path,
    output_dir: Path,
    submission_dir: Path,
    config_path: Path,
    *,
    limit: int = 100,
    dense_refine: bool = False,
    dense_top_n: int = 20,
    retrieval_texts_path: Path | None = None,
    reviewed_candidates_path: Path | None = None,
) -> dict:
    """Run all KIS files with one shared retrieval runtime and write outputs."""

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if dense_top_n < 0:
        raise ValueError("dense_top_n must be non-negative")

    query_paths = discover_kis_queries(Path(input_dir))
    submission_stems = validate_unique_submission_stems(query_paths)
    retrieval_texts = load_retrieval_texts(retrieval_texts_path, query_paths)
    reviewed_candidates = load_reviewed_candidates(reviewed_candidates_path, query_paths)
    output_root = Path(output_dir)
    submission_root = Path(submission_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    submission_root.mkdir(parents=True, exist_ok=True)
    expected_stems = {path.stem for path in query_paths}
    expected_submission_stems = set(submission_stems.values())
    removed_stale_files = [
        *_remove_stale_kis_files(output_root, expected_stems, ".json"),
        *_remove_stale_kis_files(
            submission_root,
            expected_submission_stems,
            ".csv",
        ),
    ]

    config = load_config(config_path)
    dataset_manifest_path: Path | None = None
    artifacts_root = getattr(getattr(config, "paths", None), "artifacts_root", None)
    if reviewed_candidates and artifacts_root is not None:
        dataset_manifest_path = Path(artifacts_root) / "manifests" / "dataset.json"
        validate_reviewed_candidates_against_dataset(
            reviewed_candidates,
            dataset_manifest_path,
        )
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

    # The organizer may intentionally repeat query text under distinct IDs.
    # Reuse deterministic retrieval work while still writing one file per ID.
    answer_cache: dict[str, Sequence[object]] = {}
    seen_queries: set[str] = set()
    rows: list[dict] = []
    for position, query_path in enumerate(query_paths, 1):
        query_id = query_path.stem
        query = read_query(query_path)
        retrieval_text = retrieval_texts.get(query_id, query)
        reused_query = query in seen_queries
        reused_retrieval = retrieval_text in answer_cache
        if reused_retrieval:
            answers = answer_cache[retrieval_text]
        else:
            print(
                f"[{position}/{len(query_paths)}] Retrieving {query_id}",
                file=sys.stderr,
                flush=True,
            )
            answers = tuple(solver.solve(retrieval_text))
            answer_cache[retrieval_text] = answers
        seen_queries.add(query)
        if not answers:
            raise RuntimeError(f"KIS retrieval returned no predictions for {query_id}")
        query_reviewed = reviewed_candidates.get(query_id, ())
        answers = apply_reviewed_candidates(answers, query_reviewed, limit=limit)

        payload = _result_payload(query_id, query, retrieval_text, answers)
        json_path = output_root / f"{query_id}.json"
        submission_query_id = submission_stems[query_id]
        csv_path = submission_root / f"{submission_query_id}.csv"
        atomic_write_json(json_path, payload)
        write_submission(
            csv_path,
            "kis",
            [
                RankedPrediction.kis(
                    answer.video_id,
                    answer.frame_idx,
                    score=answer.score,
                )
                for answer in answers
            ],
            include_header=False,
            max_results=limit,
        )
        top = payload["predictions"][0]
        rows.append(
            {
                "query_id": query_id,
                "submission_query_id": submission_query_id,
                "query_file": str(query_path.resolve()),
                "retrieval_text": retrieval_text,
                "retrieval_text_overridden": retrieval_text != query,
                "result_json": str(json_path.resolve()),
                "submission_csv": str(csv_path.resolve()),
                "prediction_count": len(answers),
                "reviewed_prediction_count": min(len(query_reviewed), limit),
                "reused_identical_query": reused_query,
                "reused_identical_retrieval_text": reused_retrieval,
                "top_prediction": {
                    "video_id": top["video_id"],
                    "frame_id": top["frame_ids"][0],
                    "score": top["score"],
                },
            }
        )

    manifest = {
        "schema_version": "1.0",
        "task_type": "kis",
        "input_dir": str(Path(input_dir).resolve()),
        "config": str(Path(config_path).resolve()),
        "output_dir": str(output_root.resolve()),
        "submission_dir": str(submission_root.resolve()),
        "query_count": len(rows),
        "unique_query_text_count": len({read_query(path) for path in query_paths}),
        "unique_retrieval_count": len(answer_cache),
        "retrieval_texts": (
            str(Path(retrieval_texts_path).resolve()) if retrieval_texts_path is not None else None
        ),
        "retrieval_text_override_count": sum(
            bool(row["retrieval_text_overridden"]) for row in rows
        ),
        "reviewed_candidates": (
            str(Path(reviewed_candidates_path).resolve())
            if reviewed_candidates_path is not None
            else None
        ),
        "reviewed_query_count": sum(bool(value) for value in reviewed_candidates.values()),
        "reviewed_prediction_count": sum(
            min(len(value), limit) for value in reviewed_candidates.values()
        ),
        "source_fingerprints": {
            "config": _file_fingerprint(Path(config_path)),
            "retrieval_texts": _file_fingerprint(retrieval_texts_path),
            "reviewed_candidates": _file_fingerprint(reviewed_candidates_path),
            "dataset_manifest": _file_fingerprint(dataset_manifest_path),
            "query_batch_sha256": stable_hash(
                {path.stem: file_sha256(path) for path in query_paths}
            ),
        },
        "removed_stale_files": removed_stale_files,
        "limit": limit,
        "dense_refine": dense_refine,
        "dense_top_n": dense_top_n if dense_refine else 0,
        "queries": rows,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--submission-dir", default=DEFAULT_SUBMISSION_DIR)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dense-refine", action="store_true")
    parser.add_argument("--dense-top-n", type=int, default=20)
    parser.add_argument(
        "--retrieval-texts",
        default=DEFAULT_RETRIEVAL_TEXTS,
        help=(
            "Optional UTF-8 JSON object mapping each query stem to a shorter "
            "retrieval description (for example an English CLIP prompt)."
        ),
    )
    parser.add_argument(
        "--reviewed-candidates",
        default=DEFAULT_REVIEWED_CANDIDATES,
        help=(
            "Optional UTF-8 JSON object containing verified video_id/frame_id pairs "
            "to place before automatic retrieval results."
        ),
    )
    parser.add_argument("--config", default=DEFAULT_BATCH_CONFIG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    manifest = run_batch(
        project_path(args.input_dir),
        project_path(args.output_dir),
        project_path(args.submission_dir),
        project_path(args.config),
        limit=args.limit,
        dense_refine=args.dense_refine,
        dense_top_n=args.dense_top_n,
        retrieval_texts_path=(project_path(args.retrieval_texts) if args.retrieval_texts else None),
        reviewed_candidates_path=(
            project_path(args.reviewed_candidates) if args.reviewed_candidates else None
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
