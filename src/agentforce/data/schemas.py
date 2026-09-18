"""Canonical, dependency-free records used across AgentForce.

The dataclasses in this module are deliberately boring.  They are the source of
truth exchanged between preprocessing, retrieval and evaluation, while model
vectors live outside them and are referenced by row/index identifiers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "1.0"


def utc_now_iso() -> str:
    """Return an RFC3339-ish UTC timestamp without a platform dependency."""

    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


class JsonRecord:
    """Small JSON helper shared by all canonical records."""

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass(frozen=True, slots=True)
class BoundingBox(JsonRecord):
    """Bounding box in x-min/y-min/x-max/y-max order.

    Object boxes are normalized to ``[0, 1]``. OCR backends may return pixel
    coordinates, in which case ``normalized`` is false.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    normalized: bool = True

    def __post_init__(self) -> None:
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("Bounding box maximums must be >= minimums")

    @property
    def area(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)


@dataclass(frozen=True, slots=True)
class ValidationIssue(JsonRecord):
    severity: str
    code: str
    message: str
    video_id: str | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        if self.severity not in {"info", "warning", "error"}:
            raise ValueError(f"Unsupported severity: {self.severity}")


@dataclass(slots=True)
class VideoRecord(JsonRecord):
    """One logical video and all sidecars belonging to it.

    Paths are stored relative to ``DatasetManifest.dataset_root`` whenever
    possible. A missing optional sidecar is represented by ``None``.
    """

    video_id: str
    collection: str
    split: str | None = None
    video_path: str | None = None
    keyframe_dir: str | None = None
    keyframe_map_path: str | None = None
    clip_feature_path: str | None = None
    object_dir: str | None = None
    media_info_path: str | None = None
    keyframe_count: int | None = None
    mapping_count: int | None = None
    object_frame_count: int | None = None
    feature_count: int | None = None
    feature_dimension: int | None = None
    feature_dtype: str | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    frame_count: int | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool | None = None
    title: str | None = None
    description: str | None = None
    author: str | None = None
    missing_components: list[str] = field(default_factory=list)

    @property
    def ready_for_visual_retrieval(self) -> bool:
        return all(
            (
                self.video_path,
                self.keyframe_dir,
                self.keyframe_map_path,
                self.clip_feature_path,
            )
        ) and not any(
            component in self.missing_components
            for component in ("video", "keyframes", "keyframe_map", "clip_features")
        )

    def resolve_path(self, dataset_root: str | Path, field_name: str) -> Path | None:
        value = getattr(self, field_name)
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else Path(dataset_root) / path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VideoRecord":
        return cls(**dict(data))


@dataclass(slots=True)
class DatasetManifest(JsonRecord):
    dataset_root: str
    videos: list[VideoRecord]
    issues: list[ValidationIssue] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    def by_video_id(self) -> dict[str, VideoRecord]:
        return {video.video_id: video for video in self.videos}

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> "DatasetManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            dataset_root=str(payload["dataset_root"]),
            videos=[VideoRecord.from_dict(item) for item in payload.get("videos", [])],
            issues=[ValidationIssue(**item) for item in payload.get("issues", [])],
            created_at=str(payload.get("created_at", utc_now_iso())),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class KeyframeRecord(JsonRecord):
    keyframe_uid: str
    video_id: str
    keyframe_number: int
    frame_idx: int
    pts_time: float
    fps: float
    image_path: str | None = None
    object_path: str | None = None
    visual_embedding_row: int | None = None


@dataclass(frozen=True, slots=True)
class ObjectDetection(JsonRecord):
    class_mid: str
    label: str
    score: float
    bbox: BoundingBox
    label_vi: str | None = None


@dataclass(slots=True)
class ObjectFrame(JsonRecord):
    keyframe_uid: str
    detections: list[ObjectDetection]
    counts: dict[str, int]
    searchable_text: str
    source_path: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptWord(JsonRecord):
    text: str
    start: float
    end: float
    confidence: float | None = None


@dataclass(slots=True)
class TranscriptSegment(JsonRecord):
    segment_id: int
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    confidence: float | None = None


@dataclass(slots=True)
class TranscriptDocument(JsonRecord):
    video_id: str
    language: str | None
    language_probability: float | None
    duration_seconds: float | None
    segments: list[TranscriptSegment]
    model_name: str
    source_path: str | None = None
    created_at: str = field(default_factory=utc_now_iso)

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())


@dataclass(frozen=True, slots=True)
class OCRItem(JsonRecord):
    text: str
    normalized_text: str
    confidence: float
    bbox: BoundingBox | None = None


@dataclass(slots=True)
class OCRFrame(JsonRecord):
    keyframe_uid: str
    items: list[OCRItem]
    full_text: str
    normalized_text: str
    engine: str
    source_path: str | None = None


@dataclass(slots=True)
class TemporalWindow(JsonRecord):
    window_id: str
    video_id: str
    window_number: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    representative_keyframe_uid: str | None = None
    frame_idx: int | None = None
    pts_time: float | None = None
    fps: float | None = None
    keyframe_uids: list[str] = field(default_factory=list)
    asr_segment_ids: list[int] = field(default_factory=list)
    asr_text: str = ""
    ocr_text: str = ""
    object_labels: list[str] = field(default_factory=list)
    metadata_text: str = ""


def write_jsonl(records: Iterable[JsonRecord | Mapping[str, Any]], path: str | Path) -> None:
    """Write records as UTF-8 JSON Lines using an atomic same-directory swap."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            payload = record.to_dict() if isinstance(record, JsonRecord) else _jsonable(record)
            stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def ensure_unique_ids(records: Sequence[Any], attribute: str) -> None:
    """Fail early when a canonical table would contain duplicate identifiers."""

    seen: set[Any] = set()
    for record in records:
        identifier = getattr(record, attribute)
        if identifier in seen:
            raise ValueError(f"Duplicate {attribute}: {identifier}")
        seen.add(identifier)
