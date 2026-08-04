"""Canonical dataset records and discovery utilities."""

from .layout import DatasetLayout, collection_from_video_id, is_video_id
from .manifest import build_manifest, validate_manifest
from .schemas import (
    BoundingBox,
    DatasetManifest,
    KeyframeRecord,
    ObjectDetection,
    ObjectFrame,
    OCRFrame,
    OCRItem,
    TemporalWindow,
    TranscriptDocument,
    TranscriptSegment,
    TranscriptWord,
    ValidationIssue,
    VideoRecord,
    read_jsonl,
    write_jsonl,
)
from .timeline import (
    export_timeline,
    keyframe_uid,
    keyframes_between,
    nearest_keyframe,
    read_keyframe_timeline,
    timeline_from_manifest,
)

__all__ = [
    "BoundingBox",
    "DatasetLayout",
    "DatasetManifest",
    "KeyframeRecord",
    "ObjectDetection",
    "ObjectFrame",
    "OCRFrame",
    "OCRItem",
    "TemporalWindow",
    "TranscriptDocument",
    "TranscriptSegment",
    "TranscriptWord",
    "ValidationIssue",
    "VideoRecord",
    "build_manifest",
    "collection_from_video_id",
    "export_timeline",
    "is_video_id",
    "keyframe_uid",
    "keyframes_between",
    "nearest_keyframe",
    "read_jsonl",
    "read_keyframe_timeline",
    "timeline_from_manifest",
    "validate_manifest",
    "write_jsonl",
]
