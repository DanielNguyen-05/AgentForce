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
