from __future__ import annotations

import base64
from pathlib import Path

import pytest

from agentforce.data.schemas import DatasetManifest, VideoRecord
from agentforce.visualization import (
    KeyframeRegistry,
    normalize_result_payload,
    render_gallery_html,
)


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.fixture
def registry(tmp_path: Path) -> KeyframeRegistry:
    dataset = tmp_path / "dataset"
    keyframes = dataset / "keyframes" / "L21" / "L21_V001"
    mappings = dataset / "map-keyframes"
    videos = dataset / "videos" / "L21"
    keyframes.mkdir(parents=True)
    mappings.mkdir(parents=True)
    videos.mkdir(parents=True)
    (keyframes / "001.png").write_bytes(_PNG_1X1)
    (keyframes / "002.png").write_bytes(_PNG_1X1)
    (mappings / "L21_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,1.0,25.0,25\n2,2.0,25.0,50\n",
        encoding="utf-8",
    )
    (videos / "L21_V001.mp4").write_bytes(b"fake-video")
    manifest = DatasetManifest(
        dataset_root=str(dataset),
        videos=[
            VideoRecord(
                video_id="L21_V001",
                collection="L21",
                video_path="videos/L21/L21_V001.mp4",
                keyframe_dir="keyframes/L21/L21_V001",
                keyframe_map_path="map-keyframes/L21_V001.csv",
            )
        ],
    )
    return KeyframeRegistry(manifest)


def test_search_gallery_embeds_image_and_escapes_untrusted_text(
    registry: KeyframeRegistry,
) -> None:
    payload = {
        "query": {"raw_text": "<script>alert(1)</script>", "task_type": "kis"},
        "hits": [
            {
                "rank": 1,
                "candidate_id": "L21_V001_K000001",
                "score": 0.75,
                "modality_scores": {"visual": 0.8, "asr": 0.7},
                "metadata": {
                    "video_id": "L21_V001",
                    "keyframe_uid": "L21_V001_K000001",
                    "frame_idx": 25,
                    "pts_time": 1.0,
                    "asr_text": "xin <b>chào</b>",
                    "ocr_text": "</script><script>bad()</script>",
                    "object_labels": ["person"],
                },
            }
        ],
    }

    document = normalize_result_payload(
        payload, registry, source_label="fixture.json", max_results=20
    )
    html = render_gallery_html(document)

    assert len(document.records) == 1
    assert document.records[0].media.resolution == "exact_keyframe"
    assert "data:image/jpeg;base64," in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "xin &lt;b&gt;chào&lt;/b&gt;" in html
    assert "Content-Security-Policy" in html


@pytest.mark.parametrize(
    ("task_type", "prediction", "expected_answer"),
    (
        (
            "kis",
            {
                "video_id": "L21_V001",
                "frame_ids": [25],
                "timestamp": 1.0,
                "candidate_id": "L21_V001_K000001",
                "score": 0.8,
                "evidence": {"asr_text": "xin chào"},
            },
            None,
        ),
        (
            "qa",
            {
                "video_id": "L21_V001",
                "frame_ids": [25],
                "answer": "màu đỏ",
                "score": 0.9,
                "gemini": {"confidence": 0.9},
            },
            "màu đỏ",
        ),
    ),
)
def test_kis_and_qa_predictions_are_normalized(
    registry: KeyframeRegistry,
    task_type: str,
    prediction: dict,
    expected_answer: str | None,
) -> None:
    document = normalize_result_payload(
        {"query": "query gốc", "task_type": task_type, "predictions": [prediction]},
        registry,
        source_label="task.json",
    )

    assert len(document.records) == 1
    assert document.records[0].answer == expected_answer
    assert document.records[0].media.resolution == "exact_keyframe"


def test_trake_prediction_expands_ordered_event_frames(registry: KeyframeRegistry) -> None:
    payload = {
        "query": "event một, sau đó event hai",
        "task_type": "trake",
        "predictions": [
            {
                "video_id": "L21_V001",
                "frame_ids": [25, 50],
                "event_timestamps": [1.0, 2.0],
                "event_candidate_ids": [
                    "L21_V001_K000001",
                    "L21_V001_K000002",
                ],
                "score": 0.6,
                "evidence": {
                    "event_descriptions": ["event một", "event hai"],
                    "event_scores": [0.7, 0.5],
                },
            }
        ],
    }

    document = normalize_result_payload(payload, registry, source_label="trake.json")

    assert [record.event_order for record in document.records] == [1, 2]
    assert [record.frame_idx for record in document.records] == [25, 50]
    assert [record.event_description for record in document.records] == [
        "event một",
        "event hai",
    ]


def test_dense_frame_is_marked_for_exact_video_decode_not_nearest_keyframe(
    registry: KeyframeRegistry,
) -> None:
    document = normalize_result_payload(
        {
            "task_type": "kis",
            "predictions": [
                {"video_id": "L21_V001", "frame_ids": [30], "score": 0.5}
            ],
        },
        registry,
        source_label="dense.json",
        query_override="dense result",
    )

    media = document.records[0].media
    assert media.resolution == "exact_video_frame"
    assert media.displayed_frame_idx == 30
    assert media.image_path is None
    assert media.video_path is not None


def test_trake_parallel_arrays_must_align(registry: KeyframeRegistry) -> None:
    payload = {
        "task_type": "trake",
        "predictions": [
            {
                "video_id": "L21_V001",
                "frame_ids": [25, 50],
                "event_timestamps": [1.0],
                "evidence": {},
            }
        ],
    }

    with pytest.raises(ValueError, match="must align"):
        normalize_result_payload(payload, registry, source_label="bad.json")
