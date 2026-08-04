"""JSON adapters for evaluation and submission scripts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .schemas import (
    FrameInterval,
    GroundTruth,
    KISGroundTruth,
    QAGroundTruth,
    RankedPrediction,
    TRAKEEventGroundTruth,
    TRAKEGroundTruth,
    VideoFrameInterval,
)


def parse_ground_truth(value: Mapping[str, Any]) -> GroundTruth:
    """Convert one ground-truth JSON object to its typed representation."""

    task = str(value["task_type"]).strip().lower()
    if task in {"kis", "qa"}:
        intervals = tuple(
            VideoFrameInterval.from_bounds(
                str(item["video_id"]),
                int(item["start_frame"]),
                int(item["end_frame"]),
            )
            for item in value["relevant_intervals"]
        )
        if task == "kis":
            return KISGroundTruth(intervals)
        return QAGroundTruth(intervals, tuple(str(item) for item in value["accepted_answers"]))

    if task == "trake":
        events = tuple(
            TRAKEEventGroundTruth(
                tuple(
                    FrameInterval(int(span["start_frame"]), int(span["end_frame"]))
                    for span in event["intervals"]
                )
            )
            for event in value["events"]
        )
        return TRAKEGroundTruth(str(value["video_id"]), events)

    raise ValueError(f"Unsupported ground-truth task: {task}")


def parse_prediction(value: Mapping[str, Any]) -> RankedPrediction:
    """Convert one prediction JSON object to a ranked prediction."""

    frames = value.get("frame_ids")
    if frames is None:
        frames = [value["frame_idx"]]
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ValueError("frame_ids must be an array of integers")
    return RankedPrediction(
        video_id=str(value["video_id"]),
        frame_ids=tuple(int(frame) for frame in frames),
        answer=value.get("answer"),
        score=value.get("score"),
    )


def _parse_prediction_values(value: object, *, context: str) -> tuple[RankedPrediction, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{context} must be an array of prediction objects")
    predictions: list[RankedPrediction] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{context}[{index}] must be a prediction object")
        predictions.append(parse_prediction(item))
    return tuple(predictions)


def parse_prediction_document(
    value: object,
    *,
    query_id: str | None = None,
) -> dict[str, tuple[RankedPrediction, ...]]:
    """Parse solver output, a query mapping, or a query-scoped bare list.

    Direct task scripts emit ``{query_id, task_type, predictions}``. Evaluation
    fixtures commonly use ``{query_id: [...]}``. A bare list is accepted only
    when the caller supplies ``query_id`` so predictions can never be assigned
    to an implicit query by accident.
    """

    if isinstance(value, Mapping):
        is_solver_wrapper = "predictions" in value and (
            "query_id" in value or "task_type" in value
        )
        if is_solver_wrapper:
            embedded_id = value.get("query_id")
            if not isinstance(embedded_id, str) or not embedded_id.strip():
                raise ValueError("Solver prediction output requires a non-empty query_id")
            normalized_id = embedded_id.strip()
            if query_id is not None and query_id.strip() != normalized_id:
                raise ValueError(
                    f"Requested query ID {query_id!r} does not match solver output "
                    f"query ID {normalized_id!r}"
                )
            return {
                normalized_id: _parse_prediction_values(
                    value["predictions"], context="predictions"
                )
            }

        parsed: dict[str, tuple[RankedPrediction, ...]] = {}
        for raw_id, predictions in value.items():
            normalized_id = str(raw_id).strip()
            if not normalized_id:
                raise ValueError("Prediction query IDs must not be empty")
            parsed[normalized_id] = _parse_prediction_values(
                predictions, context=f"predictions[{normalized_id!r}]"
            )
        return parsed

    if query_id is None or not query_id.strip():
        raise ValueError("query_id is required when predictions JSON is a bare array")
    normalized_id = query_id.strip()
    return {
        normalized_id: _parse_prediction_values(value, context="predictions")
    }
