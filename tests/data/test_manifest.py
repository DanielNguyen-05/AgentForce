from __future__ import annotations

import csv
import json

import numpy as np

from agentforce.data import DatasetManifest, build_manifest


def _make_video_dataset(tmp_path, *, feature_rows: int = 2):
    root = tmp_path / "dataset"
    video_id = "L26_V001"
    for directory in (
        root / "videos" / "L26_a",
        root / "keyframes" / "L26_a" / video_id,
        root / "map-keyframes",
        root / "clip-features-32",
        root / "objects" / video_id,
        root / "media-info",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (root / "videos" / "L26_a" / f"{video_id}.mp4").write_bytes(b"fake")
    for number in (1, 2):
        name = f"{number:03d}"
        (root / "keyframes" / "L26_a" / video_id / f"{name}.jpg").write_bytes(b"image")
        (root / "objects" / video_id / f"{name}.json").write_text("{}", encoding="utf-8")
    with (root / "map-keyframes" / f"{video_id}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerow({"n": 1, "pts_time": 0, "fps": 25, "frame_idx": 0})
        writer.writerow({"n": 2, "pts_time": 1, "fps": 25, "frame_idx": 25})
    np.save(
        root / "clip-features-32" / f"{video_id}.npy",
        np.ones((feature_rows, 4), dtype=np.float16),
    )
    (root / "media-info" / f"{video_id}.json").write_text(
        json.dumps({"length": 2, "title": "Video thử nghiệm", "author": "AIC"}),
        encoding="utf-8",
    )
    return root, video_id


def test_build_manifest_discovers_l26_split_and_round_trips(tmp_path) -> None:
    root, video_id = _make_video_dataset(tmp_path)
    manifest = build_manifest(root)
    assert len(manifest.videos) == 1
    video = manifest.videos[0]
    assert video.video_id == video_id
    assert video.collection == "L26"
    assert video.split == "L26_a"
    assert video.keyframe_count == video.mapping_count == video.object_frame_count == 2
    assert video.feature_count == 2
    assert video.feature_dimension == 4
    assert video.feature_dtype == "float16"
    assert video.ready_for_visual_retrieval
    assert manifest.error_count == 0
    assert manifest.warning_count == 0

    output = tmp_path / "manifest.json"
    manifest.write_json(output)
    restored = DatasetManifest.read_json(output)
    assert restored.videos[0].video_path == "videos/L26_a/L26_V001.mp4"
    assert restored.videos[0].title == "Video thử nghiệm"


def test_manifest_reports_row_count_mismatch(tmp_path) -> None:
    root, _ = _make_video_dataset(tmp_path, feature_rows=1)
    manifest = build_manifest(root)
    assert any(issue.code == "component_count_mismatch" for issue in manifest.issues)
    assert manifest.error_count == 1
