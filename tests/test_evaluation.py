from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentforce.evaluation import (
    FrameInterval,
    KISGroundTruth,
    QAGroundTruth,
    RankedPrediction,
    TaskType,
    TRAKEEventGroundTruth,
    TRAKEGroundTruth,
    VideoFrameInterval,
    evaluate_query,
    evaluate_run,
    normalize_answer,
    parse_ground_truth,
    parse_prediction,
    parse_prediction_document,
    score_prediction,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _span(video_id: str, start: int, end: int) -> VideoFrameInterval:
    return VideoFrameInterval.from_bounds(video_id, start, end)


def test_kis_uses_inclusive_frame_interval_and_rank_prefixes() -> None:
    ground_truth = KISGroundTruth((_span("L01_V001", 100, 110),))
    predictions = [RankedPrediction.kis("wrong", index) for index in range(5)]
    predictions.append(RankedPrediction.kis("L01_V001", 110))

    result = evaluate_query(ground_truth, predictions)

    assert result.recall_at_k == {1: 0.0, 5: 0.0, 20: 1.0, 50: 1.0, 100: 1.0}
    assert result.final_score == pytest.approx(0.6)


def test_qa_requires_video_frame_and_normalized_answer() -> None:
    ground_truth = QAGroundTruth(
        (_span("L01_V001", 100, 110),),
        ("Thành phố Hồ Chí Minh", "TP HCM"),
    )
    assert score_prediction(
        ground_truth,
        RankedPrediction.qa("L01_V001", 105, "  THÀNH PHỐ HỒ CHÍ MINH! "),
    ) == 1.0
    assert score_prediction(ground_truth, RankedPrediction.qa("L01_V001", 111, "TP HCM")) == 0.0
    assert score_prediction(ground_truth, RankedPrediction.qa("L01_V001", 105, "Hà Nội")) == 0.0
    assert normalize_answer(" TP. HCM ") == "tp hcm"


def test_trake_wrong_video_zero_and_correct_video_gets_event_fraction() -> None:
    ground_truth = TRAKEGroundTruth(
        video_id="L02_V002",
        events=(
            TRAKEEventGroundTruth((FrameInterval(10, 12),)),
            TRAKEEventGroundTruth((FrameInterval(20, 22),)),
            TRAKEEventGroundTruth((FrameInterval(30, 32), FrameInterval(40, 42))),
        ),
    )
    assert score_prediction(ground_truth, RankedPrediction.trake("wrong", (11, 21, 31))) == 0.0
    assert score_prediction(
        ground_truth,
        RankedPrediction.trake("L02_V002", (11, 99, 41)),
    ) == pytest.approx(2 / 3)
    assert score_prediction(ground_truth, RankedPrediction.trake("L02_V002", (11,))) == pytest.approx(1 / 3)

    result = evaluate_query(
        ground_truth,
        [
            RankedPrediction.trake("L02_V002", (11, 99, 41)),
            RankedPrediction.trake("L02_V002", (11, 21, 31)),
        ],
    )
    assert result.recall_at_k[1] == pytest.approx(2 / 3)
    assert result.recall_at_k[5] == 1.0
    assert result.final_score == pytest.approx((2 / 3 + 4) / 5)


def test_evaluate_run_macro_averages_queries_and_reports_tasks() -> None:
    ground_truths = {
        "kis": KISGroundTruth((_span("video", 1, 2),)),
        "qa": QAGroundTruth((_span("video", 3, 4),), ("đỏ",)),
    }
    predictions = {
        "kis": [RankedPrediction.kis("video", 1)],
        "qa": [RankedPrediction.qa("video", 3, "xanh")],
    }
    report = evaluate_run(ground_truths, predictions)
    assert report.final_score == 0.5
    assert report.recall_at_k[100] == 0.5
    assert report.by_task == {TaskType.KIS: 1.0, TaskType.QA: 0.0}


def test_unknown_prediction_query_id_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown query IDs"):
        evaluate_run(
            {"known": KISGroundTruth((_span("video", 1, 1),))},
            {"extra": [RankedPrediction.kis("video", 1)]},
        )


def test_json_adapters_cover_ground_truth_and_prediction_shapes() -> None:
    ground_truth = parse_ground_truth(
        {
            "task_type": "kis",
            "relevant_intervals": [
                {"video_id": "L01_V001", "start_frame": 10, "end_frame": 20}
            ],
        }
    )
    prediction = parse_prediction(
        {"video_id": "L01_V001", "frame_ids": [15], "score": 0.9}
    )
    assert isinstance(ground_truth, KISGroundTruth)
    assert score_prediction(ground_truth, prediction) == 1.0


def test_prediction_json_adapter_accepts_legacy_frame_idx() -> None:
    prediction = parse_prediction({"video_id": "L01_V001", "frame_idx": 15})
    assert prediction.frame_ids == (15,)


def test_prediction_document_accepts_direct_solver_wrapper() -> None:
    parsed = parse_prediction_document(
        {
            "query_id": "KIS001",
            "task_type": "kis",
            "predictions": [
                {"video_id": "L01_V001", "frame_ids": [15], "score": 0.9}
            ],
        }
    )
    assert parsed["KIS001"][0] == RankedPrediction.kis("L01_V001", 15, score=0.9)


def test_prediction_document_requires_identity_for_bare_array() -> None:
    with pytest.raises(ValueError, match="query_id is required"):
        parse_prediction_document([{"video_id": "L01_V001", "frame_ids": [15]}])

    parsed = parse_prediction_document(
        [{"video_id": "L01_V001", "frame_ids": [15]}],
        query_id="KIS001",
    )
    assert parsed["KIS001"][0].frame_ids == (15,)


def test_evaluation_script_accepts_direct_solver_output(tmp_path: Path) -> None:
    ground_truth = tmp_path / "ground-truth.json"
    predictions = tmp_path / "run-result.json"
    report = tmp_path / "report.json"
    ground_truth.write_text(
        json.dumps(
            {
                "KIS001": {
                    "task_type": "kis",
                    "relevant_intervals": [
                        {"video_id": "L01_V001", "start_frame": 10, "end_frame": 20}
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps(
            {
                "query_id": "KIS001",
                "task_type": "kis",
                "predictions": [{"video_id": "L01_V001", "frame_ids": [15]}],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_results.py"),
            "--ground-truth",
            str(ground_truth),
            "--predictions",
            str(predictions),
            "--output",
            str(report),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text(encoding="utf-8"))["final_score"] == 1.0
