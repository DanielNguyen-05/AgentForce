"""Offline preprocessing components with lazy optional model dependencies."""

from .asr import (
    ASRConfig,
    ASRDependencyError,
    FasterWhisperAdapter,
    TranscriptSchemaError,
    migrate_transcript,
    read_transcript,
    write_transcript,
)
from .audio import AudioExtractionConfig, FFmpegError, MediaProbe, extract_audio, probe_media
from .objects import classwise_nms, load_object_frame, normalize_object_files
from .ocr import (
    EasyOCREngine,
    OCRDependencyError,
    OCRProcessor,
    TesseractOCREngine,
    create_ocr_engine,
    normalize_ocr_text,
)
from .windows import (
    WindowConfig,
    align_transcript_to_windows,
    attach_keyframe_text,
    attach_object_labels,
    build_temporal_windows,
    interval_overlap,
)

__all__ = [
    "ASRConfig",
    "ASRDependencyError",
    "AudioExtractionConfig",
    "EasyOCREngine",
    "FFmpegError",
    "FasterWhisperAdapter",
    "TranscriptSchemaError",
    "MediaProbe",
    "OCRDependencyError",
    "OCRProcessor",
    "TesseractOCREngine",
    "WindowConfig",
    "align_transcript_to_windows",
    "attach_keyframe_text",
    "attach_object_labels",
    "build_temporal_windows",
    "classwise_nms",
    "create_ocr_engine",
    "extract_audio",
    "interval_overlap",
    "load_object_frame",
    "migrate_transcript",
    "normalize_object_files",
    "normalize_ocr_text",
    "probe_media",
    "read_transcript",
    "write_transcript",
]
