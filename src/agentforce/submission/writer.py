"""Convert ranked candidates to the task-specific CSV wire format."""

from __future__ import annotations

import csv
from io import StringIO
import os
from pathlib import Path
import tempfile
from typing import Sequence

from agentforce.evaluation.schemas import RankedPrediction, TaskType


MAX_SUBMISSION_RESULTS = 100


def _validate_predictions(
    task_type: TaskType,
    predictions: Sequence[RankedPrediction],
    max_results: int,
) -> tuple[RankedPrediction, ...]:
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= MAX_SUBMISSION_RESULTS:
        raise ValueError(f"max_results must be between 1 and {MAX_SUBMISSION_RESULTS}")
    values = tuple(predictions)
    if len(values) > max_results:
        raise ValueError(
            f"Submission has {len(values)} results, exceeding configured maximum {max_results}; "
            "rank/truncate explicitly before serialization"
        )
    trake_event_count: int | None = None
    for prediction in values:
        if not isinstance(prediction, RankedPrediction):
            raise TypeError("predictions must contain RankedPrediction values")
        if task_type in {TaskType.KIS, TaskType.QA} and len(prediction.frame_ids) != 1:
            raise ValueError(f"{task_type.value} predictions require exactly one frame ID")
        if task_type is TaskType.QA and (prediction.answer is None or not prediction.answer.strip()):
            raise ValueError("Q&A predictions require a non-empty answer")
        if task_type is not TaskType.QA and prediction.answer is not None:
            raise ValueError(f"{task_type.value} predictions must not include an answer")
        if task_type is TaskType.TRAKE:
            if trake_event_count is None:
                trake_event_count = len(prediction.frame_ids)
            elif len(prediction.frame_ids) != trake_event_count:
                raise ValueError("All TRAKE predictions for one query must have the same event count")
    return values


def submission_rows(
    task_type: TaskType | str,
    predictions: Sequence[RankedPrediction],
    *,
    max_results: int = MAX_SUBMISSION_RESULTS,
) -> list[list[str]]:
    """Return rows in rank order without a header."""

    parsed_task = TaskType.parse(task_type)
    values = _validate_predictions(parsed_task, predictions, max_results)
    rows: list[list[str]] = []
    for prediction in values:
        if parsed_task is TaskType.KIS:
            rows.append([prediction.video_id, str(prediction.frame_ids[0])])
        elif parsed_task is TaskType.QA:
            assert prediction.answer is not None  # validated above
            rows.append([prediction.video_id, str(prediction.frame_ids[0]), prediction.answer.strip()])
        else:
            rows.append([prediction.video_id, *(str(frame_idx) for frame_idx in prediction.frame_ids)])
    return rows


def _header(task_type: TaskType, rows: Sequence[Sequence[str]]) -> list[str]:
    if task_type is TaskType.KIS:
        return ["video_id", "frame_id"]
    if task_type is TaskType.QA:
        return ["video_id", "frame_id", "answer"]
    event_count = max((len(row) - 1 for row in rows), default=0)
    return ["video_id", *(f"frame_id_{index}" for index in range(1, event_count + 1))]


def serialize_submission(
    task_type: TaskType | str,
    predictions: Sequence[RankedPrediction],
    *,
    include_header: bool = False,
    max_results: int = MAX_SUBMISSION_RESULTS,
) -> str:
    """Serialize RFC-compliant CSV, including proper quoting for Q&A text."""

    parsed_task = TaskType.parse(task_type)
    rows = submission_rows(parsed_task, predictions, max_results=max_results)
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    if include_header:
        writer.writerow(_header(parsed_task, rows))
    writer.writerows(rows)
    return stream.getvalue()


def write_submission(
    path: str | Path,
    task_type: TaskType | str,
    predictions: Sequence[RankedPrediction],
    *,
    include_header: bool = False,
    max_results: int = MAX_SUBMISSION_RESULTS,
) -> Path:
    """Validate and atomically write one query's ranked submission."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = serialize_submission(
        task_type,
        predictions,
        include_header=include_header,
        max_results=max_results,
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return output
