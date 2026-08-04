"""Dataset layout discovery, including split directories such as ``L26_a``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


VIDEO_ID_PATTERN = re.compile(r"^L\d+_V\d+$", re.IGNORECASE)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


def is_video_id(value: str) -> bool:
    return VIDEO_ID_PATTERN.fullmatch(value) is not None


def collection_from_video_id(video_id: str) -> str:
    if not is_video_id(video_id):
        raise ValueError(f"Invalid video_id: {video_id!r}")
    return video_id.split("_", 1)[0].upper()


@dataclass(frozen=True, slots=True)
class DatasetLayout:
    """Paths for the organizers' dataset rooted at one directory."""

    root: Path

    @classmethod
    def from_path(cls, root: str | Path) -> "DatasetLayout":
        return cls(Path(root).expanduser().resolve())

    @property
    def videos(self) -> Path:
        return self.root / "videos"

    @property
    def keyframes(self) -> Path:
        return self.root / "keyframes"

    @property
    def keyframe_maps(self) -> Path:
        return self.root / "map-keyframes"

    @property
    def clip_features(self) -> Path:
        return self.root / "clip-features-32"

    @property
    def objects(self) -> Path:
        return self.root / "objects"

    @property
    def media_info(self) -> Path:
        return self.root / "media-info"

    def component_roots(self) -> dict[str, Path]:
        return {
            "videos": self.videos,
            "keyframes": self.keyframes,
            "keyframe_maps": self.keyframe_maps,
            "clip_features": self.clip_features,
            "objects": self.objects,
            "media_info": self.media_info,
        }

    def relative(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path.resolve())

    def iter_video_files(self) -> Iterable[Path]:
        if not self.videos.exists():
            return
        for path in sorted(self.videos.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES and is_video_id(path.stem):
                yield path

    def iter_keyframe_dirs(self) -> Iterable[Path]:
        """Yield logical keyframe directories without walking every image."""

        if not self.keyframes.exists():
            return
        candidates = [entry for entry in self.keyframes.iterdir() if entry.is_dir()]
        for candidate in sorted(candidates):
            if is_video_id(candidate.name):
                yield candidate
                continue
            for child in sorted(candidate.iterdir()):
                if child.is_dir() and is_video_id(child.name):
                    yield child

    def iter_object_dirs(self) -> Iterable[Path]:
        if not self.objects.exists():
            return
        for path in sorted(self.objects.iterdir()):
            if path.is_dir() and is_video_id(path.name):
                yield path

    def image_files(self, directory: Path) -> list[Path]:
        return sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: (int(path.stem) if path.stem.isdigit() else 10**12, path.name),
        )
