import json

import numpy as np

from agentforce.indexing.build_windows import build_visual_window_index


def test_visual_window_pooling(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "clip-features-32").mkdir(parents=True)
    (dataset / "map-keyframes").mkdir()
    np.save(
        dataset / "clip-features-32" / "L01_V001.npy",
        np.asarray([[1, 0], [0, 1]], dtype=np.float16),
    )
    (dataset / "map-keyframes" / "L01_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n2,1.0,25.0,25\n",
        encoding="utf-8",
    )
    windows = [
        {
            "window_id": "L01_V001_W000000",
            "video_id": "L01_V001",
            "keyframe_uids": ["L01_V001_K000001", "L01_V001_K000002"],
            "start_time": 0,
            "end_time": 10,
        }
    ]
    manifest = build_visual_window_index(windows, dataset, tmp_path / "out", dtype="float32")
    pooled = np.load(tmp_path / "out" / "visual_windows.npy")
    assert manifest["row_count"] == 1
    assert np.allclose(pooled[0], np.asarray([2**-0.5, 2**-0.5]), atol=1e-5)
    metadata = json.loads((tmp_path / "out" / "visual_windows.jsonl").read_text())
    assert metadata["vector_id"] == "L01_V001_W000000"
    assert metadata["frame_idx"] in {0, 25}
