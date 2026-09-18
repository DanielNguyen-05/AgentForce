"""R@1/5/20/50/100 scoring shared by offline experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Mapping, Sequence

from .schemas import GroundTruth, KISGroundTruth, QAGroundTruth, RankedPrediction, TaskType, TRAKEGroundTruth


DEFAULT_CUTOFFS: tuple[int, ...] = (1, 5, 20, 50, 100)


def normalize_answer(value: str) -> str:
    """Conservative Unicode/case/punctuation normalization.

    Diacritics and numeric meaning are preserved; semantic aliases should be
    explicitly listed in ``QAGroundTruth.accepted_answers``.
    """

    if not isinstance(value, str):
        raise TypeError("Answer must be a string")
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = "".join(" " if unicodedata.category(char)[0] in {"P", "Z"} else char for char in normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _frame_is_relevant(video_id: str, frame_idx: int, ground_truth: KISGroundTruth | QAGroundTruth) -> bool:
    return any(
        span.video_id == video_id and span.interval.contains(frame_idx)
        for span in ground_truth.relevant_intervals
    )


def score_prediction(ground_truth: GroundTruth, prediction: RankedPrediction) -> float:
    """Return correctness in ``[0, 1]`` for one ranked candidate.

    KIS and Q&A are binary. TRAKE receives partial event credit only when the
    video is correct; missing event frames count as incorrect.
    """

    if isinstance(ground_truth, KISGroundTruth):
        return float(_frame_is_relevant(prediction.video_id, prediction.frame_ids[0], ground_truth))

    if isinstance(ground_truth, QAGroundTruth):
        if not _frame_is_relevant(prediction.video_id, prediction.frame_ids[0], ground_truth):
            return 0.0
        if prediction.answer is None:
            return 0.0
        accepted = {normalize_answer(answer) for answer in ground_truth.accepted_answers}
        return float(normalize_answer(prediction.answer) in accepted)

    if isinstance(ground_truth, TRAKEGroundTruth):
        if prediction.video_id != ground_truth.video_id:
            return 0.0
        matched = sum(
            event.contains(prediction.frame_ids[index])
            for index, event in enumerate(ground_truth.events)
            if index < len(prediction.frame_ids)
        )
        return matched / len(ground_truth.events)

    raise TypeError(f"Unsupported ground truth type: {type(ground_truth)!r}")


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    values = tuple(cutoffs)
    if not values:
        raise ValueError("At least one cutoff is required")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise ValueError("Cutoffs must be positive integers")
    if len(values) != len(set(values)):
        raise ValueError("Cutoffs must be unique")
    return values


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    task_type: TaskType
    recall_at_k: Mapping[int, float]
    final_score: float
    prediction_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "task_type": self.task_type.value,
            "recall_at_k": {f"R@{cutoff}": score for cutoff, score in self.recall_at_k.items()},
            "final_score": self.final_score,
            "prediction_count": self.prediction_count,
        }


def evaluate_query(
    ground_truth: GroundTruth,
    predictions: Sequence[RankedPrediction],
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> QueryEvaluation:
    """Take the best valid candidate within each top-K prefix."""

    cutoffs = _validate_cutoffs(cutoffs)
    scores = [score_prediction(ground_truth, prediction) for prediction in predictions]
    recall_at_k = {
        cutoff: max(scores[:cutoff], default=0.0)
        for cutoff in cutoffs
    }
    final_score = math.fsum(recall_at_k.values()) / len(recall_at_k)
    return QueryEvaluation(
        task_type=ground_truth.task_type,
        recall_at_k=recall_at_k,
        final_score=final_score,
        prediction_count=len(predictions),
    )


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    recall_at_k: Mapping[int, float]
    final_score: float
    query_count: int
    by_task: Mapping[TaskType, float]
    per_query: Mapping[str, QueryEvaluation]

    def to_dict(self) -> dict[str, object]:
        return {
            "recall_at_k": {f"R@{cutoff}": score for cutoff, score in self.recall_at_k.items()},
            "final_score": self.final_score,
            "query_count": self.query_count,
            "by_task": {task.value: score for task, score in self.by_task.items()},
            "per_query": {query_id: result.to_dict() for query_id, result in self.per_query.items()},
        }


def evaluate_run(
    ground_truths: Mapping[str, GroundTruth],
    predictions: Mapping[str, Sequence[RankedPrediction]],
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> EvaluationReport:
    """Macro-average query scores; missing prediction lists score zero."""

    cutoffs = _validate_cutoffs(cutoffs)
    if not ground_truths:
        raise ValueError("At least one ground-truth query is required")
    unknown_query_ids = set(predictions).difference(ground_truths)
    if unknown_query_ids:
        raise ValueError(f"Predictions contain unknown query IDs: {sorted(unknown_query_ids)}")

    per_query = {
        query_id: evaluate_query(ground_truth, predictions.get(query_id, ()), cutoffs=cutoffs)
        for query_id, ground_truth in ground_truths.items()
    }
    query_count = len(per_query)
    recall_at_k = {
        cutoff: math.fsum(result.recall_at_k[cutoff] for result in per_query.values()) / query_count
        for cutoff in cutoffs
    }
    final_score = math.fsum(result.final_score for result in per_query.values()) / query_count

    by_task: dict[TaskType, float] = {}
    for task_type in TaskType:
        task_results = [result.final_score for result in per_query.values() if result.task_type is task_type]
        if task_results:
            by_task[task_type] = math.fsum(task_results) / len(task_results)
    return EvaluationReport(
        recall_at_k=recall_at_k,
        final_score=final_score,
        query_count=query_count,
        by_task=by_task,
        per_query=per_query,
    )
