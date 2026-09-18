from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from agentforce.config import AppConfig, PathConfig, ScopeConfig
from agentforce.data.artifact_validation import validate_artifacts
from agentforce.data.schemas import DatasetManifest, VideoRecord
from agentforce.data.scope import scope_hash
from agentforce.utils.io import canonical_jsonl_sha256, file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_IDS = ("L21_V001", "L21_V002")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_complete_artifacts(tmp_path: Path) -> AppConfig:
    dataset_root = tmp_path / "dataset"
    artifacts_root = tmp_path / "artifacts"
    outputs_root = tmp_path / "outputs"
    config_path = tmp_path / "config.toml"
    config = AppConfig(
        paths=PathConfig(
            dataset_root=dataset_root,
            artifacts_root=artifacts_root,
            outputs_root=outputs_root,
        ),
        scope=ScopeConfig(video_ids=VIDEO_IDS),
        source_path=config_path,
    )

    DatasetManifest(
        dataset_root=str(dataset_root),
        videos=[
            VideoRecord(video_id=video_id, collection="L21", keyframe_count=2)
            for video_id in VIDEO_IDS
        ],
    ).write_json(artifacts_root / "manifests" / "dataset.json")

    transcript_dir = artifacts_root / "transcripts"
    transcript_dir.mkdir(parents=True)
    for video_id in VIDEO_IDS:
        (transcript_dir / f"{video_id}.json").write_text(
            json.dumps({"source_video": f"{video_id}.mp4", "segments": []}),
            encoding="utf-8",
        )

    timeline_rows: list[dict] = []
    for video_id in VIDEO_IDS:
        for number in (1, 2):
            timeline_rows.append(
                {
                    "keyframe_uid": f"{video_id}_K{number:06d}",
                    "video_id": video_id,
                    "keyframe_number": number,
                    "frame_idx": number * 10,
                    "pts_time": float(number),
                    "fps": 25.0,
                }
            )
    _write_jsonl(artifacts_root / "timelines" / "keyframes.jsonl", timeline_rows)

    for video_id in VIDEO_IDS:
        rows = [
            {
                "keyframe_uid": f"{video_id}_K{number:06d}",
                "counts": {},
                "detections": [],
            }
            for number in (1, 2)
        ]
        _write_jsonl(artifacts_root / "objects" / f"{video_id}.jsonl", rows)
        _write_jsonl(
            artifacts_root / "ocr" / f"{video_id}.jsonl",
            [
                {
                    "keyframe_uid": f"{video_id}_K{number:06d}",
                    "items": [],
                    "normalized_text": "",
                }
                for number in (1, 2)
            ],
        )

    windows_path = artifacts_root / "windows" / "temporal_windows.jsonl"
    _write_jsonl(
        windows_path,
        [
            {
                "window_id": f"{video_id}_W000001",
                "video_id": video_id,
                "start_time": 0.0,
                "end_time": 10.0,
                "keyframe_uids": [f"{video_id}_K000001", f"{video_id}_K000002"],
            }
            for video_id in VIDEO_IDS
        ],
    )
    window_hash = canonical_jsonl_sha256(windows_path, exclude_fields=("metadata_text",))
    windows_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "name": "temporal_windows",
                "path": str(windows_path),
                "sha256": file_sha256(windows_path),
                "canonical_records_hash": window_hash,
                "video_count": len(VIDEO_IDS),
                "video_ids": list(VIDEO_IDS),
                "scope_hash": scope_hash(VIDEO_IDS),
            }
        ),
        encoding="utf-8",
    )

    index_dir = artifacts_root / "indexes"
    index_dir.mkdir(parents=True)
    for name in ("visual_keyframes", "visual_windows", "asr_windows", "objects_windows"):
        np.save(index_dir / f"{name}.npy", np.ones((2, 3), dtype=np.float32))
        _write_jsonl(
            index_dir / f"{name}.jsonl",
            [
                {"vector_id": f"{video_id}-1", "video_id": video_id}
                for video_id in VIDEO_IDS
            ],
        )
        manifest = {
            "name": name,
            "vectors_path": f"{name}.npy",
            "metadata_path": f"{name}.jsonl",
            "row_count": 2,
            "dimension": 3,
            "dtype": "float32",
            "video_ids": list(VIDEO_IDS),
            "scope_hash": scope_hash(VIDEO_IDS),
        }
        if name.endswith("_windows"):
            manifest["window_records_hash"] = window_hash
        (index_dir / f"{name}.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return config


def _write_config(config: AppConfig) -> Path:
    assert config.source_path is not None
    config.source_path.write_text(
        "\n".join(
            (
                "[paths]",
                f"dataset_root = {json.dumps(str(config.paths.dataset_root))}",
                f"artifacts_root = {json.dumps(str(config.paths.artifacts_root))}",
                f"outputs_root = {json.dumps(str(config.paths.outputs_root))}",
                "",
                "[scope]",
                f"video_ids = {json.dumps(list(VIDEO_IDS))}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return config.source_path


def test_complete_strict_artifact_set_is_valid(tmp_path: Path) -> None:
    report = validate_artifacts(_build_complete_artifacts(tmp_path), strict=True)
    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.to_dict()["ok"] is True


def test_ocr_is_warning_unless_explicitly_required(tmp_path: Path) -> None:
    config = _build_complete_artifacts(tmp_path)
    (config.paths.artifacts_root / "ocr" / "L21_V002.jsonl").unlink()

    optional = validate_artifacts(config, strict=True, require_ocr=False)
    required = validate_artifacts(config, strict=True, require_ocr=True)

    assert optional.error_count == 0
    assert any(issue.code == "missing_ocr" for issue in optional.issues)
    assert all(
        issue.severity == "warning" for issue in optional.issues if issue.code == "missing_ocr"
    )
    assert any(
        issue.code == "missing_ocr" and issue.severity == "error"
        for issue in required.issues
    )


def test_caption_artifacts_and_bad_index_shape_are_errors(tmp_path: Path) -> None:
    config = _build_complete_artifacts(tmp_path)
    caption_dir = config.paths.artifacts_root / "captions"
    caption_dir.mkdir()
    (caption_dir / "stale.jsonl").write_text("{}\n", encoding="utf-8")
    index_manifest = config.paths.artifacts_root / "indexes" / "visual_keyframes.manifest.json"
    payload = json.loads(index_manifest.read_text(encoding="utf-8"))
    payload["dimension"] = 99
    index_manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_artifacts(config, strict=True)
    codes = {issue.code for issue in report.issues}

    assert "caption_artifacts_forbidden" in codes
    assert "index_vector_dimension_mismatch" in codes


def test_transcript_window_scope_and_hash_drift_are_errors(tmp_path: Path) -> None:
    config = _build_complete_artifacts(tmp_path)
    transcript = config.paths.transcripts_dir / "L21_V002.json"
    transcript.write_text(
        json.dumps({"source_video": "L21_V999.mp4", "segments": []}),
        encoding="utf-8",
    )
    windows = config.paths.artifacts_root / "windows" / "temporal_windows.jsonl"
    with windows.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"window_id": "bad", "video_id": "L21_V999"}) + "\n")

    report = validate_artifacts(config, strict=True)
    codes = {issue.code for issue in report.issues}

    assert "transcript_video_id_mismatch" in codes
    assert "window_row_scope_mismatch" in codes
    assert "window_file_hash_mismatch" in codes
    assert "window_canonical_hash_mismatch" in codes


def test_strict_mode_requires_visual_asr_and_object_indexes(tmp_path: Path) -> None:
    config = _build_complete_artifacts(tmp_path)
    index_dir = config.paths.artifacts_root / "indexes"
    for suffix in (".manifest.json", ".npy", ".jsonl"):
        (index_dir / f"asr_windows{suffix}").unlink()

    relaxed = validate_artifacts(config, strict=False)
    strict = validate_artifacts(config, strict=True)

    assert not any(issue.code == "missing_required_index" for issue in relaxed.issues)
    assert any(
        issue.code == "missing_required_index" and "asr_windows" in issue.message
        for issue in strict.issues
    )


def test_script_prints_report_and_returns_two_for_strict_errors(tmp_path: Path) -> None:
    config = _build_complete_artifacts(tmp_path)
    config_path = _write_config(config)
    report_path = tmp_path / "report.json"
    caption_dir = config.paths.artifacts_root / "captions"
    caption_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validate_artifacts.py"),
            "--config",
            str(config_path),
            "--strict",
            "--output",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "[RUN]" in result.stderr
    stdout_report = json.loads(result.stdout)
    stored_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout_report == stored_report
    assert stdout_report["errors"] >= 1
    assert any(
        issue["code"] == "caption_artifacts_forbidden"
        for issue in stdout_report["issues"]
    )
