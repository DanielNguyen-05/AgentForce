import importlib.util

import pytest

from agentforce.temporal.opencv_source import OpenCVFrameSource


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="opencv is optional")
def test_opencv_source_requires_known_video(tmp_path) -> None:
    source = OpenCVFrameSource({"L01_V001": tmp_path / "missing.mp4"})
    with pytest.raises(FileNotFoundError):
        list(source.iter_frames("L01_V001", start_time=0, end_time=1, sample_fps=1))
