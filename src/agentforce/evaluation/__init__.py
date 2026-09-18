"""Official-style metrics for AIC KIS, Q&A and TRAKE tasks."""

from .metrics import (
    DEFAULT_CUTOFFS,
    EvaluationReport,
    QueryEvaluation,
    evaluate_query,
    evaluate_run,
    normalize_answer,
    score_prediction,
)
from .io import parse_ground_truth, parse_prediction, parse_prediction_document
from .schemas import (
    FrameInterval,
    KISGroundTruth,
    QAGroundTruth,
    RankedPrediction,
    TaskType,
    TRAKEEventGroundTruth,
    TRAKEGroundTruth,
    VideoFrameInterval,
)

__all__ = [
    "DEFAULT_CUTOFFS",
    "EvaluationReport",
    "FrameInterval",
    "KISGroundTruth",
    "QAGroundTruth",
    "QueryEvaluation",
    "RankedPrediction",
    "TaskType",
    "TRAKEEventGroundTruth",
    "TRAKEGroundTruth",
    "VideoFrameInterval",
    "evaluate_query",
    "evaluate_run",
    "normalize_answer",
    "parse_ground_truth",
    "parse_prediction",
    "parse_prediction_document",
    "score_prediction",
]
