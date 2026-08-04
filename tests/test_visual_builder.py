import csv
import json

import numpy as np
import pytest

from agentforce.data.scope import scope_hash
from agentforce.indexing.build_visual import build_visual_index, inspect_visual_shards
from agentforce.indexing.numpy_index import NumpyIndex


def _write_shard(dataset, video_id: str, features: np.ndarray) -> None:
    (dataset / "clip-features-32").mkdir(parents=True, exist_ok=True)
    (dataset / "map-keyframes").mkdir(exist_ok=True)
    np.save(dataset / "clip-features-32" / f"{video_id}.npy", features)
    with (dataset / "map-keyframes" / f"{video_id}.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        for row in range(features.shape[0]):
            writer.writerow(
                {
                    "n": row + 1,
                    "pts_time": row,
                    "fps": 25,
                    "frame_idx": row * 25,
                }
            )


def test_build_visual_index(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    _write_shard(dataset, "L01_V001", np.eye(2, dtype=np.float16))

    expected_encoder = {"model": "custom-clip", "pretrained": "organizer-v1"}
    manifest = build_visual_index(
        dataset,
        tmp_path / "out",
        expected_encoder=expected_encoder,
    )
    assert manifest["row_count"] == 2
    assert manifest["video_ids"] == ["L01_V001"]
    assert manifest["scope_hash"] == scope_hash(["L01_V001"])
    assert manifest["expected_encoder"] == expected_encoder
    persisted = json.loads(
        (tmp_path / "out" / "visual_keyframes.manifest.json").read_text()
    )
    assert persisted == manifest
    assert not list((tmp_path / "out").glob(".*.tmp"))
    index = NumpyIndex(
        tmp_path / "out" / "visual_keyframes.npy",
        tmp_path / "out" / "visual_keyframes.jsonl",
    )
    assert index.search(np.asarray([0, 1]), 1)[0].metadata["frame_idx"] == 25


def test_visual_index_scope_is_explicit_sorted_and_deterministic(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    _write_shard(dataset, "L01_V001", np.asarray([[1.0, 0.0]], dtype=np.float16))
    _write_shard(
        dataset,
        "L01_V002",
        np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=np.float16),
    )
    _write_shard(dataset, "L01_V003", np.asarray([[0.5, 0.5]], dtype=np.float16))

    assert inspect_visual_shards(
        dataset, video_ids=["L01_V002", "L01_V001"]
    ) == (3, 2)
    first = build_visual_index(
        dataset,
        tmp_path / "first",
        video_ids=["L01_V002", "L01_V001"],
    )
    second = build_visual_index(
        dataset,
        tmp_path / "second",
        video_ids=["l01_v001", "l01_v002"],
    )
    full = build_visual_index(dataset, tmp_path / "full")

    assert first["video_ids"] == ["L01_V001", "L01_V002"]
    assert first["scope_hash"] == second["scope_hash"]
    assert first["source_signature"] == second["source_signature"]
    assert first["source_signature"] != full["source_signature"]
    assert first["expected_encoder"] is None
    metadata = [
        json.loads(line)
        for line in (tmp_path / "first" / "visual_keyframes.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["video_id"] for row in metadata] == [
        "L01_V001",
        "L01_V002",
        "L01_V002",
    ]


@pytest.mark.parametrize(
    ("video_ids", "error", "message"),
    [
        (["L01_V001", "l01_v001"], ValueError, "Duplicate video IDs"),
        (["L01_V999"], FileNotFoundError, "L01_V999"),
        ([], ValueError, "at least one video ID"),
    ],
)
def test_visual_scope_rejects_invalid_ids(tmp_path, video_ids, error, message) -> None:
    dataset = tmp_path / "dataset"
    _write_shard(dataset, "L01_V001", np.asarray([[1.0, 0.0]], dtype=np.float16))

    with pytest.raises(error, match=message):
        inspect_visual_shards(dataset, video_ids=video_ids)


def test_visual_builder_cleans_temporary_files_before_atomic_commit(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    _write_shard(dataset, "L01_V001", np.asarray([[1.0, 0.0]], dtype=np.float16))
    (dataset / "map-keyframes" / "L01_V001.csv").unlink()
    output = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match="Missing keyframe mapping"):
        build_visual_index(dataset, output, video_ids=["L01_V001"])

    assert list(output.iterdir()) == []
