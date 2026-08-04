"""OpenCV implementation of exact frame decoding around coarse timestamps."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from agentforce.errors import OptionalDependencyError
from .refinement import DecodedFrame


class OpenCVFrameSource:
    """Decode sparse or dense canonical frames without loading a whole video."""

    def __init__(
        self,
        video_paths: Mapping[str, str | Path] | None = None,
        *,
        resolver: Callable[[str], str | Path] | None = None,
    ) -> None:
        if video_paths is None and resolver is None:
            raise ValueError("Provide video_paths or a resolver")
        self._paths = {key: Path(value) for key, value in (video_paths or {}).items()}
        self._resolver = resolver

    def _path(self, video_id: str) -> Path:
        value = self._paths.get(video_id)
        if value is None and self._resolver is not None:
            value = Path(self._resolver(video_id))
        if value is None:
            raise KeyError(f"No video path for {video_id}")
        if not value.is_file():
            raise FileNotFoundError(value)
        return value

    def iter_frames(
        self,
        video_id: str,
        *,
        start_time: float,
        end_time: float,
        sample_fps: float,
    ) -> Iterable[DecodedFrame]:
        if start_time < 0 or end_time < start_time or sample_fps <= 0:
            raise ValueError("Expected 0 <= start_time <= end_time and sample_fps > 0")
        try:
            import cv2
        except ImportError as exc:
            raise OptionalDependencyError(
                "Dense frame decoding requires `pip install -e '.[video]'`"
            ) from exc

        path = self._path(video_id)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"OpenCV cannot open {path}")
        try:
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if source_fps <= 0:
                raise RuntimeError(f"Video has invalid FPS: {path}")
            start_frame = max(0, round(start_time * source_fps))
            end_frame = max(start_frame, round(end_time * source_fps))
            step = max(1, round(source_fps / sample_fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            while frame_idx <= end_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                if actual < start_frame:
                    actual = frame_idx
                if (actual - start_frame) % step == 0:
                    yield DecodedFrame(
                        video_id=video_id,
                        frame_idx=actual,
                        timestamp=actual / source_fps,
                        payload=frame,
                        metadata={
                            "video_path": str(path),
                            "fps": source_fps,
                            "color_space": "bgr",
                        },
                    )
                frame_idx = actual + 1
        finally:
            capture.release()

