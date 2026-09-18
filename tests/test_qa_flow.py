from __future__ import annotations

import csv
from pathlib import Path

from agentforce.data.schemas import DatasetManifest, VideoRecord
from agentforce.gemini import QAVerification, QAVerificationRequest
from agentforce.retrieval.types import FusedHit
from agentforce.tasks.qa import QAConfig, QASolver


class RecordingVerifier:
    """Network-free verifier that records the exact local frames it receives."""

    def __init__(self) -> None:
        self.calls: list[QAVerificationRequest] = []

    def verify(self, request: QAVerificationRequest) -> QAVerification:
        self.calls.append(request)
        supporting_frame = request.candidates[0]
        result = QAVerification(
            answerable=True,
            answer=f"answer-{supporting_frame.candidate_id}",
            normalized_answer=f"answer-{supporting_frame.candidate_id}",
            answer_type="object",
            supporting_candidate_id=supporting_frame.candidate_id,
            confidence=0.9,
            evidence=("Visible in the selected local frame.",),
        )
        return result.with_local_support(supporting_frame, cache_hit=False)


def _manifest_and_hits(
    tmp_path: Path,
    *,
    count: int = 5,
) -> tuple[DatasetManifest, list[FusedHit]]:
    video_id = "L21_V001"
    keyframes = tmp_path / "keyframes" / "L21" / video_id
    objects = tmp_path / "objects" / video_id
    mappings = tmp_path / "map-keyframes"
    keyframes.mkdir(parents=True)
    objects.mkdir(parents=True)
    mappings.mkdir()

    with (mappings / f"{video_id}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        for number in range(1, count + 1):
            (keyframes / f"{number:03d}.jpg").write_bytes(f"image-{number}".encode())
            writer.writerow(
                {
                    "n": number,
                    "pts_time": number * 2,
                    "fps": 25,
                    "frame_idx": number * 50,
                }
            )

    manifest = DatasetManifest(
        dataset_root=str(tmp_path),
        videos=[
            VideoRecord(
                video_id=video_id,
                collection="L21",
                keyframe_dir=f"keyframes/L21/{video_id}",
                keyframe_map_path=f"map-keyframes/{video_id}.csv",
                object_dir=f"objects/{video_id}",
            )
        ],
    )
    hits = [
        FusedHit(
            "window",
            1.0 - number / 100,
            metadata={
                "video_id": video_id,
                "representative_keyframe_uid": f"{video_id}_K{number:06d}",
            },
        )
        for number in range(1, count + 1)
    ]
    return manifest, hits


def test_ranked_competition_flow_uses_only_disjoint_locally_resolved_batches(
    tmp_path: Path,
) -> None:
    manifest, hits = _manifest_and_hits(tmp_path)
    verifier = RecordingVerifier()
    solver = QASolver(
        verifier,  # type: ignore[arg-type]
        manifest,
        config=QAConfig(max_candidates=5, batch_size=2),
    )

    answers = solver.solve_ranked(
        query_id="QA001",
        question="Vật thể có màu gì?",
        retrieval_context="một người đang cầm vật thể",
        candidates=hits,
    )

    assert [call.query_id for call in verifier.calls] == ["QA001-B01", "QA001-B02", "QA001-B03"]
    assert [[frame.candidate_id for frame in call.candidates] for call in verifier.calls] == [
        ["C01", "C02"],
        ["C03", "C04"],
        ["C05"],
    ]
    assert len(answers) == 3
    all_frames = [frame for call in verifier.calls for frame in call.candidates]
    assert len({(frame.video_id, frame.frame_idx) for frame in all_frames}) == 5
    assert all(frame.image_path.is_file() for frame in all_frames)
    assert all(tmp_path in frame.image_path.parents for frame in all_frames)


def test_single_answer_smoke_flow_makes_exactly_one_grounded_call(tmp_path: Path) -> None:
    manifest, hits = _manifest_and_hits(tmp_path)
    verifier = RecordingVerifier()
    solver = QASolver(
        verifier,  # type: ignore[arg-type]
        manifest,
        config=QAConfig(max_candidates=5, batch_size=2),
    )

    result = solver.solve(
        query_id="QA001",
        question="Vật thể có màu gì?",
        retrieval_context="một người đang cầm vật thể",
        candidates=hits,
    )

    assert len(verifier.calls) == 1
    assert len(verifier.calls[0].candidates) == 5
    assert result.supporting_video_id == "L21_V001"
    assert result.supporting_frame_idx == 50
