import csv

from agentforce.data.schemas import DatasetManifest, VideoRecord
from agentforce.retrieval.types import FusedHit
from agentforce.tasks.qa import QACandidateBuilder


def test_qa_candidate_builder_resolves_trusted_local_frame(tmp_path) -> None:
    keyframes = tmp_path / "keyframes" / "L01" / "L01_V001"
    objects = tmp_path / "objects" / "L01_V001"
    maps = tmp_path / "map-keyframes"
    keyframes.mkdir(parents=True)
    objects.mkdir(parents=True)
    maps.mkdir()
    (keyframes / "001.jpg").write_bytes(b"image")
    with (maps / "L01_V001.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerow({"n": 1, "pts_time": 2, "fps": 25, "frame_idx": 50})
    manifest = DatasetManifest(
        dataset_root=str(tmp_path),
        videos=[
            VideoRecord(
                video_id="L01_V001",
                collection="L01",
                keyframe_dir="keyframes/L01/L01_V001",
                keyframe_map_path="map-keyframes/L01_V001.csv",
                object_dir="objects/L01_V001",
            )
        ],
    )
    hit = FusedHit(
        "window",
        0.8,
        metadata={
            "video_id": "L01_V001",
            "representative_keyframe_uid": "L01_V001_K000001",
        },
    )
    frames = QACandidateBuilder(manifest).build([hit], limit=5)
    assert frames[0].frame_idx == 50
    assert frames[0].image_path == keyframes / "001.jpg"
