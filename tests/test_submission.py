from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentforce.evaluation import RankedPrediction, TaskType
from agentforce.submission import serialize_submission, submission_rows, write_submission

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_kis_submission_preserves_rank_order() -> None:
    predictions = [
        RankedPrediction.kis("L01_V001", 10),
        RankedPrediction.kis("L01_V002", 20),
    ]
    assert submission_rows(TaskType.KIS, predictions) == [
        ["L01_V001", "10"],
        ["L01_V002", "20"],
    ]
    assert serialize_submission("kis", predictions, include_header=True) == (
        "video_id,frame_id\nL01_V001,10\nL01_V002,20\n"
    )


def test_qa_csv_quotes_answers_with_commas() -> None:
    content = serialize_submission(
        TaskType.QA,
        [RankedPrediction.qa("L01_V001", 10, "đỏ, xanh")],
    )
    assert content == 'L01_V001,10,"đỏ, xanh"\n'


def test_trake_serializes_ordered_event_frames_and_dynamic_header() -> None:
    prediction = RankedPrediction.trake("L02_V002", (10, 20, 30))
    assert serialize_submission(TaskType.TRAKE, [prediction], include_header=True) == (
        "video_id,frame_id_1,frame_id_2,frame_id_3\nL02_V002,10,20,30\n"
    )


def test_submission_does_not_silently_truncate() -> None:
    predictions = [RankedPrediction.kis("video", index) for index in range(3)]
    with pytest.raises(ValueError, match="exceeding configured maximum"):
        serialize_submission(TaskType.KIS, predictions, max_results=2)


def test_writer_is_utf8_and_validates_task_fields(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "qa.csv"
    write_submission(
        output,
        TaskType.QA,
        [RankedPrediction.qa("L01_V001", 10, "màu đỏ")],
    )
    assert output.read_text(encoding="utf-8") == "L01_V001,10,màu đỏ\n"

    with pytest.raises(ValueError, match="require a non-empty answer"):
        serialize_submission(TaskType.QA, [RankedPrediction.kis("L01_V001", 10)])


def test_submission_script_accepts_direct_solver_output(tmp_path: Path) -> None:
    source = tmp_path / "run-result.json"
    output = tmp_path / "submission.csv"
    source.write_text(
        json.dumps(
            {
                "query_id": "KIS001",
                "task_type": "kis",
                "predictions": [
                    {"video_id": "L01_V001", "frame_ids": [15], "score": 0.9}
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "write_submission.py"),
            "--task",
            "kis",
            "--predictions",
            str(source),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == "L01_V001,15\n"


def test_submission_script_rejects_solver_task_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "run-result.json"
    source.write_text(
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
            str(PROJECT_ROOT / "scripts" / "write_submission.py"),
            "--task",
            "qa",
            "--predictions",
            str(source),
            "--output",
            str(tmp_path / "submission.csv"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "does not match prediction output task" in result.stderr
