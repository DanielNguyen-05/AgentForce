import json

import numpy as np

from agentforce.embeddings.encoders import HashingTextEncoder
from agentforce.engine import (
    IndexTextSearcher,
    MultimodalSearchEngine,
    SearchField,
    canonical_fusion_id,
)
from agentforce.indexing.numpy_index import NumpyIndex
from agentforce.tasks import KISConfig, KISSolver


def test_multimodal_engine_returns_submission_metadata(tmp_path) -> None:
    encoder = HashingTextEncoder(16)
    vectors = encoder.encode(["người mở laptop", "vận động viên đạp xe"])
    np.save(tmp_path / "vectors.npy", vectors)
    with (tmp_path / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"vector_id": "A", "video_id": "L01_V001", "frame_idx": 50, "pts_time": 2}
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {"vector_id": "B", "video_id": "L01_V002", "frame_idx": 75, "pts_time": 3}
            )
            + "\n"
        )
    index = NumpyIndex(tmp_path / "vectors.npy", tmp_path / "metadata.jsonl")
    engine = MultimodalSearchEngine([SearchField("ocr", index, encoder, top_k=2)])
    result = engine.search("người mở laptop", task_type="kis")
    assert result.hits[0].candidate_id == "A"
    assert result.hits[0].metadata["frame_idx"] == 50


def _write_index(tmp_path, name, vectors, rows) -> NumpyIndex:
    vectors_path = tmp_path / f"{name}.npy"
    metadata_path = tmp_path / f"{name}.jsonl"
    np.save(vectors_path, vectors)
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return NumpyIndex(vectors_path, metadata_path)


def test_canonical_fusion_id_priority() -> None:
    assert canonical_fusion_id(
        "L21_V001_W000001",
        {
            "representative_keyframe_uid": "L21_V001_K000010",
            "keyframe_uid": "L21_V001_K000009",
        },
    ) == "L21_V001_K000010"
    assert canonical_fusion_id(
        "L21_V001_W000001",
        {"representative_keyframe_uid": "", "keyframe_uid": "L21_V001_K000009"},
    ) == "L21_V001_K000009"
    assert canonical_fusion_id("L21_V001_W000001", {}) == "L21_V001_W000001"


def test_visual_and_asr_fuse_on_representative_keyframe_with_source_context(tmp_path) -> None:
    encoder = HashingTextEncoder(32)
    query_vector = encoder.encode(["người đang mở laptop"])
    keyframe_uid = "L21_V001_K000010"
    window_id = "L21_V001_W000004"
    visual = _write_index(
        tmp_path,
        "visual",
        query_vector,
        [
            {
                "vector_id": keyframe_uid,
                "keyframe_uid": keyframe_uid,
                "video_id": "L21_V001",
                "frame_idx": 250,
                "pts_time": 10.0,
            }
        ],
    )
    asr = _write_index(
        tmp_path,
        "asr",
        query_vector,
        [
            {
                "vector_id": window_id,
                "window_id": window_id,
                "representative_keyframe_uid": keyframe_uid,
                "video_id": "L21_V001",
                "frame_idx": 250,
                "pts_time": 10.0,
                "text": "người đang mở laptop",
                "asr_text": "người đang mở laptop",
            }
        ],
    )
    fields = [
        SearchField("visual", visual, encoder, top_k=1),
        SearchField("asr", asr, encoder, top_k=1),
    ]

    result = MultimodalSearchEngine(fields, rank_constant=0).search(
        "người đang mở laptop",
        task_type="kis",
    )

    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.candidate_id == keyframe_uid
    assert set(hit.modality_scores) == {"visual", "asr"}
    assert hit.metadata["visual_vector_id"] == keyframe_uid
    assert hit.metadata["asr_vector_id"] == window_id
    assert hit.metadata["window_id"] == window_id
    assert hit.metadata["asr_text"] == "người đang mở laptop"


def test_index_text_searchers_keep_task_adapters_working_after_canonicalization(tmp_path) -> None:
    encoder = HashingTextEncoder(32)
    query_vector = encoder.encode(["người đang mở laptop"])
    keyframe_uid = "L21_V001_K000010"
    window_id = "L21_V001_W000004"
    visual = _write_index(
        tmp_path,
        "task_visual",
        query_vector,
        [
            {
                "vector_id": keyframe_uid,
                "keyframe_uid": keyframe_uid,
                "video_id": "L21_V001",
                "frame_idx": 250,
                "pts_time": 10.0,
            }
        ],
    )
    asr = _write_index(
        tmp_path,
        "task_asr",
        query_vector,
        [
            {
                "vector_id": window_id,
                "window_id": window_id,
                "representative_keyframe_uid": keyframe_uid,
                "video_id": "L21_V001",
                "frame_idx": 250,
                "pts_time": 10.0,
                "asr_text": "người đang mở laptop",
            }
        ],
    )
    searchers = {
        "visual": IndexTextSearcher(SearchField("visual", visual, encoder, top_k=1)),
        "asr": IndexTextSearcher(SearchField("asr", asr, encoder, top_k=1)),
    }

    answers = KISSolver(
        searchers,
        config=KISConfig(per_source_k=1, fusion_pool_size=5, top_k=5),
    ).solve("người đang mở laptop")

    assert len(answers) == 1
    assert answers[0].candidate_id == keyframe_uid
    assert answers[0].video_id == "L21_V001"
    assert answers[0].frame_idx == 250
    assert set(answers[0].evidence["modality_scores"]) == {
        "visual:original:0",
        "asr:original:0",
    }
    assert answers[0].evidence["asr_vector_id"] == window_id


def test_adjacent_windows_deduplicate_only_when_their_representative_is_identical(
    tmp_path,
) -> None:
    encoder = HashingTextEncoder(32)
    query_vector = encoder.encode(["người đang mở laptop"])[0]
    first_keyframe = "L21_V001_K000010"
    next_keyframe = "L21_V001_K000011"
    asr = _write_index(
        tmp_path,
        "adjacent_asr",
        np.vstack([query_vector, query_vector * 0.8, query_vector * 0.7]),
        [
            {
                "vector_id": "L21_V001_W000004",
                "window_id": "L21_V001_W000004",
                "representative_keyframe_uid": first_keyframe,
                "video_id": "L21_V001",
                "frame_idx": 250,
                "pts_time": 10.0,
            },
            {
                "vector_id": "L21_V001_W000005",
                "window_id": "L21_V001_W000005",
                "representative_keyframe_uid": first_keyframe,
                "video_id": "L21_V001",
                "frame_idx": 250,
                "pts_time": 10.0,
            },
            {
                "vector_id": "L21_V001_W000006",
                "window_id": "L21_V001_W000006",
                "representative_keyframe_uid": next_keyframe,
                "video_id": "L21_V001",
                "frame_idx": 275,
                "pts_time": 11.0,
            },
        ],
    )

    result = MultimodalSearchEngine(
        [SearchField("asr", asr, encoder, top_k=3)],
        duplicate_seconds=2.0,
    ).search("người đang mở laptop", task_type="kis")

    assert {hit.candidate_id for hit in result.hits} == {first_keyframe, next_keyframe}
    first = next(hit for hit in result.hits if hit.candidate_id == first_keyframe)
    assert first.metadata["window_id"] == "L21_V001_W000004"
    assert first.metadata["asr_vector_id"] == "L21_V001_W000004"
