from __future__ import annotations

import csv

from agentforce.data.timeline import nearest_keyframe, read_keyframe_timeline


def test_timeline_joins_numbered_sidecars_and_uses_zero_based_vector_rows(tmp_path) -> None:
    mapping = tmp_path / "L26_V001.csv"
    images = tmp_path / "images"
    objects = tmp_path / "objects"
    images.mkdir()
    objects.mkdir()
    for number in (1, 2):
        (images / f"{number:03d}.jpg").write_bytes(b"image")
        (objects / f"{number:03d}.json").write_text("{}", encoding="utf-8")
    with mapping.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerow({"n": 1, "pts_time": 0.0, "fps": 25, "frame_idx": 0})
        writer.writerow({"n": 2, "pts_time": 0.84, "fps": 25, "frame_idx": 21})

    timeline = read_keyframe_timeline(
        mapping, "L26_V001", image_dir=images, object_dir=objects, require_sidecars=True
    )
    assert timeline[0].keyframe_uid == "L26_V001_K000001"
    assert timeline[0].visual_embedding_row == 0
    assert timeline[1].frame_idx == 21
    assert nearest_keyframe(timeline, 0.6) == timeline[1]

