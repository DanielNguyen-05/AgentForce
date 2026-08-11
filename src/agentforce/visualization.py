"""Normalize retrieval/task results and render a safe standalone keyframe gallery."""

from __future__ import annotations

import base64
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cache
from html import escape
from importlib import resources
from io import BytesIO
import json
import mimetypes
from pathlib import Path
from typing import Any

from agentforce.data.schemas import DatasetManifest, KeyframeRecord
from agentforce.data.timeline import nearest_keyframe, timeline_from_manifest


@dataclass(frozen=True, slots=True)
class ResolvedMedia:
    resolution: str
    image_path: Path | None = None
    video_path: Path | None = None
    displayed_frame_idx: int | None = None
    displayed_timestamp: float | None = None
    keyframe_uid: str | None = None
    delta_frames: int | None = None
    delta_seconds: float | None = None


@dataclass(slots=True)
class GalleryRecord:
    record_id: str
    source_kind: str
    task_type: str
    prediction_rank: int
    video_id: str
    frame_idx: int | None
    timestamp: float | None
    score: float | None
    group_score: float | None
    candidate_id: str | None
    event_order: int | None = None
    event_description: str | None = None
    answer: str | None = None
    asr_text: str = ""
    ocr_text: str = ""
    object_labels: tuple[str, ...] = ()
    modality_scores: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    media: ResolvedMedia = field(default_factory=lambda: ResolvedMedia("missing"))


@dataclass(slots=True)
class GalleryDocument:
    query_text: str
    task_type: str
    query_id: str | None
    source_label: str
    records: list[GalleryRecord]
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class KeyframeRegistry:
    """Resolve result identifiers only through the canonical dataset manifest."""

    def __init__(self, manifest: DatasetManifest) -> None:
        self.manifest = manifest
        self._videos = manifest.by_video_id()
        self._timelines: dict[str, list[KeyframeRecord]] = {}
        self._uid_maps: dict[str, dict[str, KeyframeRecord]] = {}
        self._frame_maps: dict[str, dict[int, KeyframeRecord]] = {}
        self._dataset_root = Path(manifest.dataset_root).expanduser().resolve()
        self._keyframe_root = (self._dataset_root / "keyframes").resolve()
        self._video_root = (self._dataset_root / "videos").resolve()

    @staticmethod
    def _inside(path: Path, root: Path) -> Path:
        resolved = path.expanduser().resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Resolved media path escapes canonical dataset root: {resolved}") from exc
        return resolved

    def _load(self, video_id: str) -> list[KeyframeRecord]:
        if video_id not in self._videos:
            raise ValueError(f"Result references unknown video_id: {video_id}")
        if video_id not in self._timelines:
            timeline = timeline_from_manifest(self.manifest, video_id)
            self._timelines[video_id] = timeline
            self._uid_maps[video_id] = {record.keyframe_uid: record for record in timeline}
            self._frame_maps[video_id] = {record.frame_idx: record for record in timeline}
        return self._timelines[video_id]

    def _keyframe_media(
        self,
        record: KeyframeRecord,
        *,
        resolution: str = "exact_keyframe",
        requested_frame_idx: int | None = None,
        requested_timestamp: float | None = None,
    ) -> ResolvedMedia:
        image = (
            self._inside(Path(record.image_path), self._keyframe_root)
            if record.image_path is not None
            else None
        )
        return ResolvedMedia(
            resolution=resolution if image is not None else "missing_image",
            image_path=image,
            displayed_frame_idx=record.frame_idx,
            displayed_timestamp=record.pts_time,
            keyframe_uid=record.keyframe_uid,
            delta_frames=(
                None if requested_frame_idx is None else record.frame_idx - requested_frame_idx
            ),
            delta_seconds=(
                None if requested_timestamp is None else record.pts_time - requested_timestamp
            ),
        )

    def resolve(
        self,
        *,
        video_id: str,
        candidate_id: str | None,
        keyframe_uid: str | None,
        frame_idx: int | None,
        timestamp: float | None,
    ) -> ResolvedMedia:
        timeline = self._load(video_id)
        uid = keyframe_uid
        if uid is None and candidate_id and "_K" in candidate_id:
            uid = candidate_id
        uid_record = self._uid_maps[video_id].get(uid) if uid else None

        if uid_record is not None and (
            frame_idx is None or uid_record.frame_idx == frame_idx
        ):
            return self._keyframe_media(uid_record)

        if frame_idx is not None:
            exact = self._frame_maps[video_id].get(frame_idx)
            if exact is not None:
                return self._keyframe_media(exact)
            video = self._videos[video_id]
            video_path = video.resolve_path(self._dataset_root, "video_path")
            if video_path is not None:
                safe_video = self._inside(video_path, self._video_root)
                return ResolvedMedia(
                    resolution="exact_video_frame",
                    video_path=safe_video,
                    displayed_frame_idx=frame_idx,
                    displayed_timestamp=timestamp,
                    keyframe_uid=uid_record.keyframe_uid if uid_record is not None else None,
                )

        if uid_record is not None:
            return self._keyframe_media(
                uid_record,
                resolution="nearest_keyframe",
                requested_frame_idx=frame_idx,
                requested_timestamp=timestamp,
            )

        if timestamp is not None:
            nearest = nearest_keyframe(timeline, timestamp)
            if nearest is not None:
                return self._keyframe_media(
                    nearest,
                    resolution="nearest_keyframe",
                    requested_frame_idx=frame_idx,
                    requested_timestamp=timestamp,
                )
        return ResolvedMedia("missing")


def _mapping(value: object, location: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a JSON object")
    return dict(value)


def _list(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a JSON array")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object, location: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{location} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be an integer") from exc


def _optional_float(value: object, location: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{location} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be numeric") from exc


def _query_text(payload: Mapping[str, Any], override: str | None) -> str:
    if override and override.strip():
        return override.strip()
    query = payload.get("query")
    if isinstance(query, Mapping):
        for key in ("raw_text", "retrieval_text", "question"):
            value = _optional_str(query.get(key))
            if value:
                return value
    value = _optional_str(query)
    if value:
        return value
    context = _optional_str(payload.get("retrieval_context"))
    question = _optional_str(payload.get("question"))
    if context and question and context != question:
        return f"{context} — {question}"
    return context or question or _optional_str(payload.get("query_id")) or "(không có query)"


def _object_labels(*sources: Mapping[str, Any]) -> tuple[str, ...]:
    for source in sources:
        value = source.get("object_labels")
        if isinstance(value, list):
            return tuple(str(item) for item in value if str(item).strip())
        text = source.get("objects_text")
        if isinstance(text, str) and text.strip():
            return tuple(item.strip() for item in text.split(",") if item.strip())
    return ()


def _contexts(*sources: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    asr = ""
    ocr = ""
    for source in sources:
        if not asr and isinstance(source.get("asr_text"), str):
            asr = str(source["asr_text"]).strip()
        if not ocr and isinstance(source.get("ocr_text"), str):
            ocr = str(source["ocr_text"]).strip()
    return asr, ocr, _object_labels(*sources)


def _modality_scores(*sources: Mapping[str, Any]) -> dict[str, float]:
    for source in sources:
        raw = source.get("modality_scores")
        if not isinstance(raw, Mapping):
            continue
        scores: dict[str, float] = {}
        for key, value in raw.items():
            score = _optional_float(value, f"modality_scores.{key}")
            if score is not None:
                scores[str(key).split(":", 1)[0]] = score
        return scores
    return {}


def _resolve_record_media(
    registry: KeyframeRegistry,
    *,
    video_id: str,
    candidate_id: str | None,
    metadata: Mapping[str, Any],
    frame_idx: int | None,
    timestamp: float | None,
) -> ResolvedMedia:
    uid = _optional_str(
        metadata.get("keyframe_uid") or metadata.get("representative_keyframe_uid")
    )
    return registry.resolve(
        video_id=video_id,
        candidate_id=candidate_id,
        keyframe_uid=uid,
        frame_idx=frame_idx,
        timestamp=timestamp,
    )


def normalize_result_payload(
    payload: Mapping[str, Any],
    registry: KeyframeRegistry,
    *,
    source_label: str,
    query_override: str | None = None,
    max_results: int = 20,
) -> GalleryDocument:
    """Normalize run_search/KIS/QA/TRAKE JSON into gallery records."""

    if max_results < 1:
        raise ValueError("max_results must be positive")
    root = dict(payload)
    query_text = _query_text(root, query_override)
    query_id = _optional_str(root.get("query_id"))
    query_object = root.get("query")
    inferred_task = (
        query_object.get("task_type") if isinstance(query_object, Mapping) else None
    )
    task_type = _optional_str(root.get("task_type") or inferred_task) or "unknown"
    records: list[GalleryRecord] = []

    if "hits" in root:
        hits = _list(root["hits"], "hits")
        for fallback_rank, raw_hit in enumerate(hits[:max_results], 1):
            hit = _mapping(raw_hit, f"hits[{fallback_rank - 1}]")
            metadata = _mapping(hit.get("metadata"), f"hits[{fallback_rank - 1}].metadata")
            video_id = _optional_str(metadata.get("video_id"))
            if video_id is None:
                raise ValueError(f"hits[{fallback_rank - 1}] has no video_id")
            rank = _optional_int(hit.get("rank"), "hit.rank") or fallback_rank
            candidate_id = _optional_str(hit.get("candidate_id"))
            frame_idx = _optional_int(
                metadata.get("frame_idx", metadata.get("frame_id")), "hit.frame_idx"
            )
            timestamp = _optional_float(
                metadata.get("pts_time", metadata.get("timestamp")), "hit.timestamp"
            )
            asr, ocr, objects = _contexts(metadata)
            score = _optional_float(hit.get("score"), "hit.score")
            records.append(
                GalleryRecord(
                    record_id=f"search:p{rank}:e0",
                    source_kind="search_hit",
                    task_type=task_type,
                    prediction_rank=rank,
                    video_id=video_id,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    score=score,
                    group_score=score,
                    candidate_id=candidate_id,
                    asr_text=asr,
                    ocr_text=ocr,
                    object_labels=objects,
                    modality_scores=_modality_scores(hit, metadata),
                    evidence={"metadata": metadata},
                    media=_resolve_record_media(
                        registry,
                        video_id=video_id,
                        candidate_id=candidate_id,
                        metadata=metadata,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                    ),
                )
            )
    elif "predictions" in root:
        predictions = _list(root["predictions"], "predictions")
        for rank, raw_prediction in enumerate(predictions[:max_results], 1):
            prediction = _mapping(raw_prediction, f"predictions[{rank - 1}]")
            video_id = _optional_str(prediction.get("video_id"))
            if video_id is None:
                raise ValueError(f"predictions[{rank - 1}] has no video_id")
            frames = _list(prediction.get("frame_ids"), f"predictions[{rank - 1}].frame_ids")
            if not frames:
                raise ValueError(f"predictions[{rank - 1}].frame_ids must not be empty")
            evidence = _mapping(
                prediction.get("evidence"), f"predictions[{rank - 1}].evidence"
            )
            overall_score = _optional_float(prediction.get("score"), "prediction.score")
            if task_type == "trake" or len(frames) > 1:
                timestamps = prediction.get("event_timestamps")
                candidate_ids = prediction.get("event_candidate_ids")
                descriptions = evidence.get("event_descriptions")
                event_scores = evidence.get("event_scores")
                parallel = (
                    ("event_timestamps", timestamps),
                    ("event_candidate_ids", candidate_ids),
                    ("event_descriptions", descriptions),
                    ("event_scores", event_scores),
                )
                for name, values in parallel:
                    if values is not None and (
                        not isinstance(values, list) or len(values) != len(frames)
                    ):
                        raise ValueError(
                            f"predictions[{rank - 1}].{name} must align with frame_ids"
                        )
                for event_index, raw_frame in enumerate(frames, 1):
                    frame_idx = _optional_int(raw_frame, "prediction.frame_ids[]")
                    timestamp = _optional_float(
                        timestamps[event_index - 1] if isinstance(timestamps, list) else None,
                        "prediction.event_timestamp",
                    )
                    candidate_id = _optional_str(
                        candidate_ids[event_index - 1]
                        if isinstance(candidate_ids, list)
                        else None
                    )
                    event_score = _optional_float(
                        event_scores[event_index - 1]
                        if isinstance(event_scores, list)
                        else overall_score,
                        "prediction.event_score",
                    )
                    description = _optional_str(
                        descriptions[event_index - 1]
                        if isinstance(descriptions, list)
                        else None
                    )
                    records.append(
                        GalleryRecord(
                            record_id=f"{task_type}:p{rank}:e{event_index}",
                            source_kind="prediction",
                            task_type=task_type,
                            prediction_rank=rank,
                            event_order=event_index,
                            event_description=description,
                            video_id=video_id,
                            frame_idx=frame_idx,
                            timestamp=timestamp,
                            score=event_score,
                            group_score=overall_score,
                            candidate_id=candidate_id,
                            evidence=evidence,
                            media=_resolve_record_media(
                                registry,
                                video_id=video_id,
                                candidate_id=candidate_id,
                                metadata=evidence,
                                frame_idx=frame_idx,
                                timestamp=timestamp,
                            ),
                        )
                    )
                continue

            frame_idx = _optional_int(frames[0], "prediction.frame_ids[0]")
            candidate_id = _optional_str(prediction.get("candidate_id"))
            timestamp = _optional_float(
                prediction.get("timestamp", evidence.get("pts_time")),
                "prediction.timestamp",
            )
            gemini = _mapping(prediction.get("gemini"), "prediction.gemini")
            asr, ocr, objects = _contexts(evidence, gemini)
            records.append(
                GalleryRecord(
                    record_id=f"{task_type}:p{rank}:e0",
                    source_kind="prediction",
                    task_type=task_type,
                    prediction_rank=rank,
                    video_id=video_id,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    score=overall_score,
                    group_score=overall_score,
                    candidate_id=candidate_id,
                    answer=_optional_str(prediction.get("answer")),
                    asr_text=asr,
                    ocr_text=ocr,
                    object_labels=objects,
                    modality_scores=_modality_scores(prediction, evidence),
                    evidence={"evidence": evidence, "gemini": gemini},
                    media=_resolve_record_media(
                        registry,
                        video_id=video_id,
                        candidate_id=candidate_id,
                        metadata=evidence,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                    ),
                )
            )
    else:
        raise ValueError("Unsupported result JSON: expected top-level hits or predictions")

    return GalleryDocument(
        query_text=query_text,
        task_type=task_type,
        query_id=query_id,
        source_label=source_label,
        records=records,
    )


def _format_timestamp(value: float | None) -> str:
    if value is None:
        return "—"
    total_milliseconds = max(0, round(value * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _image_from_path(path: Path, *, width: int, quality: int) -> tuple[str | None, str | None]:
    if not path.is_file():
        return None, f"Không tìm thấy ảnh: {path}"
    try:
        from PIL import Image, ImageOps
    except ImportError:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        try:
            return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}", None
        except OSError as exc:
            return None, str(exc)
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if image.width > width:
                height = max(1, round(image.height * width / image.width))
                image.thumbnail((width, height), Image.Resampling.LANCZOS)
            stream = BytesIO()
            image.save(stream, format="JPEG", quality=quality, optimize=True)
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}", None
    except (OSError, ValueError) as exc:
        return None, f"Không đọc được ảnh {path}: {exc}"


def _image_from_video(
    path: Path,
    frame_idx: int,
    *,
    width: int,
    quality: int,
) -> tuple[str | None, str | None]:
    try:
        import cv2
    except ImportError:
        return None, "Cần cài project extra 'video' để decode dense frame"
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None, f"Không mở được video: {path}"
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, image = capture.read()
        if not ok or image is None:
            return None, f"Không decode được frame {frame_idx} từ {path.name}"
        height, current_width = image.shape[:2]
        if current_width > width:
            target_height = max(1, round(height * width / current_width))
            image = cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            return None, f"Không encode được frame {frame_idx} từ {path.name}"
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}", None
    finally:
        capture.release()


def _record_image(
    record: GalleryRecord,
    *,
    width: int,
    quality: int,
) -> tuple[str | None, str | None]:
    if record.media.image_path is not None:
        return _image_from_path(record.media.image_path, width=width, quality=quality)
    if record.media.video_path is not None and record.frame_idx is not None:
        return _image_from_video(
            record.media.video_path,
            record.frame_idx,
            width=width,
            quality=quality,
        )
    return None, "Không resolve được media canonical cho kết quả này"


def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _badge(text: str, css_class: str = "") -> str:
    return f'<span class="badge {css_class}">{escape(text)}</span>'


@cache
def _design_tokens_css() -> str:
    """Shared design tokens (DESIGN_SYSTEM.md), inlined because the gallery CSP forbids assets."""
    tokens = resources.files("agentforce.assets").joinpath("design_tokens.css")
    return tokens.read_text(encoding="utf-8")


_ICON_SEARCH = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>'
)
_ICON_IMAGE_OFF = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<line x1="2" y1="2" x2="22" y2="22"/><path d="M10.41 10.41a2 2 0 1 1-2.83-2.83"/>'
    '<line x1="13.5" y1="13.5" x2="6" y2="21"/><line x1="18" y1="12" x2="21" y2="15"/>'
    '<path d="M3.59 3.59A1.99 1.99 0 0 0 3 5v14a2 2 0 0 0 2 2h14c.55 0 1.052-.22 1.41-.59"/>'
    '<path d="M21 15V5a2 2 0 0 0-2-2H9"/></svg>'
)
_ICON_EMPTY = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="m13.5 8.5-5 5"/><path d="m8.5 8.5 5 5"/><circle cx="11" cy="11" r="8"/>'
    '<path d="m21 21-4.3-4.3"/></svg>'
)

# Gallery-specific styles on top of the shared tokens. __COLUMNS__ is substituted at render time.
_GALLERY_CSS = """
/* ---- Page header ---- */
.page-header { max-width: 1800px; margin: auto; padding: var(--space-5) var(--space-6) 0;
  animation: slideInUp var(--duration-slow) var(--ease-out) both; }
.title-row { display: flex; align-items: center; gap: var(--space-2_5); flex-wrap: wrap; }
.task-chip { background: var(--primary); color: var(--primary-foreground);
  font: 600 var(--text-2xs)/1 var(--font-sans); letter-spacing: 0.05em; text-transform: uppercase;
  padding: var(--space-1_5) var(--space-2_5); border-radius: var(--radius-full);
  box-shadow: var(--shadow-primary); }
h1 { margin: 0; font-size: var(--text-xl); font-weight: 700; line-height: 1.2;
  overflow-wrap: anywhere; }
.query { margin: var(--space-1) 0 0; font-size: var(--text-sm); color: var(--muted-foreground);
  max-width: 90ch; overflow-wrap: anywhere; }
.summary { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2);
  margin-top: var(--space-3); }
.stat-chip { display: inline-flex; align-items: center; gap: var(--space-1);
  background: var(--secondary); color: var(--muted-foreground); border-radius: var(--radius-full);
  padding: var(--space-1_5) var(--space-3); font-size: var(--text-xs); white-space: nowrap; }
.stat-chip b { color: var(--foreground); font-weight: 600; }

/* ---- Sticky filter toolbar ---- */
.toolbar { position: sticky; top: 0; z-index: 10; margin-top: var(--space-4);
  padding: var(--space-3) var(--space-6);
  background: color-mix(in srgb, var(--background) 86%, transparent);
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border); }
.toolbar-inner { max-width: 1800px; margin: auto; display: grid;
  grid-template-columns: minmax(220px, 1fr) repeat(3, minmax(150px, 190px));
  gap: var(--space-2); }
.search-box { position: relative; min-width: 0; }
.search-box svg { position: absolute; left: var(--space-3); top: 50%; width: 16px; height: 16px;
  translate: 0 -50%; color: var(--muted-foreground); pointer-events: none; }
.search-box .input { padding-left: 36px; }
.select-wrap { position: relative; min-width: 0; }
.select-wrap::after { content: ""; position: absolute; right: var(--space-3); top: 50%;
  width: 7px; height: 7px; margin-top: -5px; pointer-events: none;
  border-right: 1.5px solid var(--muted-foreground);
  border-bottom: 1.5px solid var(--muted-foreground); transform: rotate(45deg); }
.select { appearance: none; -webkit-appearance: none; padding-right: 30px; cursor: pointer; }

/* ---- Results grid ---- */
main { max-width: 1800px; margin: auto; padding: var(--space-5) var(--space-6) var(--space-6);
  display: grid; grid-template-columns: repeat(__COLUMNS__, minmax(0, 1fr));
  gap: var(--space-4); align-items: start; }
.result-group { min-width: 0; animation: fadeIn var(--duration-fast) var(--ease-out) both; }
.result-group.multi { grid-column: 1 / -1; }
.cards { display: grid; grid-template-columns: 1fr; gap: var(--space-4); }
.result-group.multi .cards { grid-template-columns: repeat(__COLUMNS__, minmax(0, 1fr)); }

/* ---- Result card ---- */
.card { padding: 0; overflow: hidden; display: flex; flex-direction: column; min-width: 0; }
.card-top { display: flex; justify-content: space-between; align-items: center;
  gap: var(--space-2); padding: var(--space-3) var(--space-4); }
.rank-pill { background: var(--primary); color: var(--primary-foreground);
  border-radius: var(--radius-full); font: 700 var(--text-xs)/1 var(--font-sans);
  padding: var(--space-1) var(--space-2_5); white-space: nowrap; }
.group-score { font: 500 11px var(--font-mono); color: var(--muted-foreground); }
.card-head { display: flex; flex-direction: column; align-items: flex-start;
  gap: var(--space-1_5); padding: 0 var(--space-4) var(--space-3); }
.card-title { font-size: var(--text-sm); font-weight: 600; overflow-wrap: anywhere; }
.badges { display: flex; flex-wrap: wrap; gap: var(--space-1); }
.badges .badge { font-size: var(--text-2xs); padding: 3px var(--space-2); }
.badge[class*="mod-"]::before { content: ""; width: 6px; height: 6px;
  border-radius: var(--radius-full); background: currentColor; flex: none; }
.mod-visual { color: var(--mod-visual); }
.mod-asr { color: var(--mod-asr); }
.mod-ocr { color: var(--mod-ocr); }
.mod-objects { color: var(--mod-objects); }
.badge.object { margin: 0 var(--space-1) var(--space-1) 0; }

.image-link { display: block; background: #05070a; outline-offset: -3px; }
.image-link img { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; }
.missing { aspect-ratio: 16 / 9; display: grid; place-items: center; align-content: center;
  gap: var(--space-2); padding: var(--space-5); text-align: center;
  background: var(--muted); color: var(--muted-foreground); font-size: var(--text-xs); }
.missing svg { width: 24px; height: 24px; color: var(--status-warning-fg); }

.meta { display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-2_5) var(--space-4);
  padding: var(--space-3) var(--space-4); border-top: 1px solid var(--border); }
.meta-item { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.meta-label { font: 500 var(--text-2xs) var(--font-sans); text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--muted-foreground); }
.meta-value { font: 500 13px var(--font-mono); overflow-wrap: anywhere; }

.actions { display: flex; justify-content: space-between; align-items: center;
  gap: var(--space-2); padding: 0 var(--space-4) var(--space-3); }
.btn-sm { height: 30px; padding: 0 var(--space-2_5); font-size: var(--text-xs);
  border: 1px solid var(--input); background: transparent; box-shadow: none; }
.btn-sm:hover { background: var(--secondary); }
.path { color: var(--accent); font-size: var(--text-xs); font-weight: 500;
  text-decoration: none; }
.path:hover { text-decoration: underline; }

/* ---- Expandable evidence ---- */
details.context { border-top: 1px solid var(--border); margin-top: auto;
  background: color-mix(in srgb, var(--muted) 40%, transparent); }
details.context summary { cursor: pointer; list-style: none; display: flex; align-items: center;
  gap: var(--space-2); padding: var(--space-3) var(--space-4);
  font: 500 var(--text-xs) var(--font-sans); color: var(--muted-foreground);
  transition: color var(--duration-fast); }
details.context summary::-webkit-details-marker { display: none; }
details.context summary:hover { color: var(--foreground); }
details.context summary::before { content: ""; width: 6px; height: 6px; flex: none;
  border-right: 1.5px solid currentColor; border-bottom: 1.5px solid currentColor;
  transform: rotate(-45deg); transition: transform var(--duration-fast); }
details.context[open] > summary::before { transform: rotate(45deg); }
.context-body { padding: 0 var(--space-4) var(--space-4); }
.context-body h4 { margin: var(--space-3) 0 var(--space-1_5);
  font: 600 var(--text-2xs) var(--font-sans); text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--muted-foreground); }
.context-body p { margin: 0; font-size: 13px; white-space: pre-wrap; overflow-wrap: anywhere; }
.answer { font-weight: 600; color: var(--accent); font-size: var(--text-sm); }
pre { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 280px; overflow: auto;
  margin: var(--space-1_5) 0 0; padding: var(--space-3); border-radius: var(--radius-sm);
  background: var(--muted); font: 11px var(--font-mono); }
.context-body details summary { cursor: pointer; color: var(--muted-foreground);
  font: 500 var(--text-2xs) var(--font-sans); text-transform: uppercase;
  letter-spacing: 0.05em; margin-top: var(--space-3); }
.context-body details summary:hover { color: var(--foreground); }
.scores { width: 100%; border-collapse: collapse; font-size: var(--text-xs); }
.scores td { padding: var(--space-1_5) var(--space-2); border-bottom: 1px solid var(--border); }
.scores tr:last-child td { border-bottom: 0; }
.scores td:last-child { text-align: right; font-family: var(--font-mono); }

/* ---- Empty state & pagination ---- */
.empty { grid-column: 1 / -1; display: grid; place-items: center; gap: var(--space-2);
  padding: 64px var(--space-6); text-align: center; color: var(--muted-foreground);
  background: var(--card); border: 1px dashed var(--border); border-radius: var(--radius-xl); }
.empty svg { width: 32px; height: 32px; }
.empty strong { color: var(--foreground); font-size: var(--text-sm); }
.hidden { display: none; }

.pagination { display: flex; justify-content: center; align-items: center; gap: var(--space-1_5);
  flex-wrap: wrap; padding: var(--space-2) var(--space-6) 48px; }
.pagination button { min-width: 36px; height: 36px; padding: 0 var(--space-2_5);
  border-radius: var(--radius-md); border: 1px solid var(--border); background: var(--card);
  color: var(--foreground); font: 500 13px var(--font-sans); cursor: pointer;
  box-shadow: var(--shadow-xs); transition: all var(--duration-fast); }
.pagination button:hover:not([disabled]):not(.active) { background: var(--secondary); }
.pagination button[disabled] { opacity: 0.5; cursor: default; }
.pagination button.active { background: var(--primary); border-color: var(--primary);
  color: var(--primary-foreground); box-shadow: var(--shadow-primary); }
.pagination .gap { color: var(--muted-foreground); padding: 0 2px; }
.pagination .page-info { color: var(--muted-foreground); font-size: var(--text-xs);
  margin-left: var(--space-2); }

/* ---- Focus & responsive ---- */
button:focus-visible, a:focus-visible, summary:focus-visible {
  outline: none; border-radius: var(--radius-sm);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 50%, transparent); }
@media (min-width: 768px) { h1 { font-size: var(--text-2xl); } }
@media (min-width: 1024px) { h1 { font-size: var(--text-3xl); } }
@media (max-width: 1023px) {
  main, .result-group.multi .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 767px) {
  .toolbar-inner { grid-template-columns: 1fr 1fr; }
  .search-box { grid-column: 1 / -1; }
}
@media (max-width: 639px) {
  .page-header, .toolbar, main { padding-left: var(--space-3); padding-right: var(--space-3); }
  .pagination { padding-left: var(--space-3); padding-right: var(--space-3); }
  main, .result-group.multi .cards { grid-template-columns: 1fr; }
  .toolbar { position: static; }
  .toolbar-inner { grid-template-columns: 1fr; }
}
"""


def _record_card(
    record: GalleryRecord,
    *,
    image_src: str | None,
    image_error: str | None,
    rank: int,
    group_score: float | None,
) -> str:
    event = (
        f"Event {record.event_order}: {record.event_description or 'không có mô tả'}"
        if record.event_order is not None
        else None
    )
    media_label = {
        "exact_keyframe": "exact keyframe",
        "exact_video_frame": "exact video frame",
        "nearest_keyframe": "nearest keyframe",
        "missing_image": "missing image",
        "missing": "missing",
    }.get(record.media.resolution, record.media.resolution)
    media_badge_class = (
        "badge-warning" if record.media.resolution != "exact_keyframe" else "badge-success"
    )
    badges = [
        _badge(media_label, media_badge_class),
        *(_badge(name, f"badge-tag mod-{name}") for name in sorted(record.modality_scores)),
    ]
    image_html = (
        f'<a class="image-link" href="{image_src}" target="_blank" rel="noopener">'
        f'<img loading="lazy" src="{image_src}" alt="{escape(record.record_id, quote=True)}"></a>'
        if image_src
        else f'<div class="missing">{_ICON_IMAGE_OFF}'
        f"<span>{escape(image_error or 'Không có ảnh')}</span></div>"
    )
    modality_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{score:.6f}</td></tr>"
        for name, score in sorted(record.modality_scores.items())
    )
    modality_html = (
        f'<table class="scores"><tbody>{modality_rows}</tbody></table>'
        if modality_rows
        else '<span class="muted">Không có modality score</span>'
    )
    context_parts: list[str] = []
    if record.answer:
        context_parts.append(
            f'<section><h4>Câu trả lời QA</h4><p class="answer">{escape(record.answer)}</p></section>'
        )
    if record.asr_text:
        context_parts.append(f"<section><h4>ASR</h4><p>{escape(record.asr_text)}</p></section>")
    if record.ocr_text:
        context_parts.append(f"<section><h4>OCR</h4><p>{escape(record.ocr_text)}</p></section>")
    if record.object_labels:
        objects = "".join(_badge(label, "badge-tag object") for label in record.object_labels)
        context_parts.append(f"<section><h4>Objects</h4><div>{objects}</div></section>")
    evidence = escape(
        json.dumps(record.evidence, ensure_ascii=False, indent=2, default=str)
    )
    context_parts.append(
        f"<section><h4>Modality scores</h4>{modality_html}</section>"
        f"<details><summary>Raw evidence</summary><pre>{evidence}</pre></details>"
    )
    requested = record.frame_idx if record.frame_idx is not None else "—"
    displayed = record.media.displayed_frame_idx
    display_note = ""
    if displayed is not None and displayed != record.frame_idx:
        display_note = f" · ảnh hiển thị frame {displayed}"
    original_path = record.media.image_path or record.media.video_path
    original_link = (
        f'<a class="path" href="{escape(original_path.as_uri(), quote=True)}">mở media gốc</a>'
        if original_path is not None and original_path.exists()
        else ""
    )
    title = escape(event or record.candidate_id or record.record_id)
    return f"""
      <article class="card" data-video="{escape(record.video_id, quote=True)}">
        <div class="card-top"><span class="rank-pill">Rank #{rank}</span><span class="group-score">score {_score(group_score)}</span></div>
        <div class="card-head"><strong class="card-title">{title}</strong><div class="badges">{''.join(badges)}</div></div>
        {image_html}
        <div class="meta">
          <div class="meta-item"><span class="meta-label">Video</span><span class="meta-value">{escape(record.video_id)}</span></div>
          <div class="meta-item"><span class="meta-label">Frame</span><span class="meta-value">{requested}{escape(display_note)}</span></div>
          <div class="meta-item"><span class="meta-label">Time</span><span class="meta-value">{_format_timestamp(record.timestamp)}</span></div>
          <div class="meta-item"><span class="meta-label">Score</span><span class="meta-value">{_score(record.score)}</span></div>
        </div>
        <div class="actions">
          <button type="button" class="btn btn-sm" data-copy="{escape(f'{record.video_id},{requested}', quote=True)}">copy video,frame</button>
          {original_link}
        </div>
        <details class="context"><summary>Evidence chi tiết</summary><div class="context-body">{''.join(context_parts)}</div></details>
      </article>
    """


def render_gallery_html(
    document: GalleryDocument,
    *,
    columns: int = 3,
    thumbnail_width: int = 640,
    jpeg_quality: int = 82,
) -> str:
    """Return one self-contained HTML document with embedded thumbnails."""

    if columns < 1 or columns > 8:
        raise ValueError("columns must be between 1 and 8")
    if thumbnail_width < 160 or thumbnail_width > 1920:
        raise ValueError("thumbnail_width must be between 160 and 1920")
    if jpeg_quality < 30 or jpeg_quality > 95:
        raise ValueError("jpeg_quality must be between 30 and 95")

    grouped: dict[int, list[GalleryRecord]] = defaultdict(list)
    for record in document.records:
        grouped[record.prediction_rank].append(record)
    image_cache: dict[tuple[object, ...], tuple[str | None, str | None]] = {}
    groups_html: list[str] = []
    for rank, records in grouped.items():
        group_score = records[0].group_score
        cards: list[str] = []
        for record in records:
            cache_key = (
                record.media.image_path,
                record.media.video_path,
                record.frame_idx,
                thumbnail_width,
                jpeg_quality,
            )
            if cache_key not in image_cache:
                image_cache[cache_key] = _record_image(
                    record, width=thumbnail_width, quality=jpeg_quality
                )
            image_src, image_error = image_cache[cache_key]
            cards.append(
                _record_card(
                    record,
                    image_src=image_src,
                    image_error=image_error,
                    rank=rank,
                    group_score=group_score,
                )
            )
        videos = sorted({record.video_id for record in records})
        searchable = " ".join(
            (
                document.query_text,
                *videos,
                *(record.candidate_id or "" for record in records),
                *(record.event_description or "" for record in records),
                *(record.answer or "" for record in records),
                *(record.asr_text for record in records),
                *(record.ocr_text for record in records),
                *(" ".join(record.object_labels) for record in records),
            )
        ).casefold()
        group_class = "result-group multi" if len(cards) > 1 else "result-group"
        groups_html.append(
            f"""
            <section class="{group_class}" data-rank="{rank}" data-score="{group_score or 0.0}"
              data-video="{escape(' '.join(videos), quote=True)}"
              data-search="{escape(searchable, quote=True)}">
              <div class="cards">{''.join(cards)}</div>
            </section>
            """
        )
    videos = sorted({record.video_id for record in document.records})
    video_options = "".join(
        f'<option value="{escape(video, quote=True)}">{escape(video)}</option>'
        for video in videos
    )
    empty_message = (
        ""
        if document.records
        else f'<div class="empty">{_ICON_EMPTY}'
        "<strong>Không có kết quả</strong>"
        "<span>Không có frame nào trong result JSON.</span></div>"
    )
    query_id = document.query_id or "—"
    title = f"{document.task_type.upper()} · {query_id}"
    styles = _design_tokens_css() + _GALLERY_CSS.replace("__COLUMNS__", str(columns))
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; object-src 'none'; base-uri 'none'">
  <title>{escape(title)}</title>
  <style>{styles}</style>
</head>
<body>
  <header class="page-header">
    <div class="title-row">
      <span class="task-chip">{escape(document.task_type.upper())}</span>
      <h1>{escape(query_id if document.query_id else "Kết quả truy vấn")}</h1>
    </div>
    <p class="query">{escape(document.query_text)}</p>
    <div class="summary">
      <span class="stat-chip"><b>{len(grouped)}</b> ranked results</span>
      <span class="stat-chip"><b>{len(document.records)}</b> frames</span>
      <span class="stat-chip">nguồn <b>{escape(document.source_label)}</b></span>
      <span class="stat-chip">tạo lúc <b>{escape(document.generated_at)}</b></span>
    </div>
  </header>
  <div class="toolbar">
    <div class="toolbar-inner">
      <div class="search-box">{_ICON_SEARCH}
        <input id="filter" class="input" type="search" placeholder="Lọc ASR, OCR, object, candidate...">
      </div>
      <div class="select-wrap"><select id="video" class="select"><option value="">tất cả video</option>{video_options}</select></div>
      <div class="select-wrap"><select id="sort" class="select"><option value="rank">xếp theo rank</option><option value="score">xếp theo score</option><option value="video">xếp theo video</option></select></div>
      <div class="select-wrap"><select id="page-size" class="select"><option value="5">5 kết quả/trang</option><option value="10" selected>10 kết quả/trang</option><option value="20">20 kết quả/trang</option><option value="all">tất cả</option></select></div>
    </div>
  </div>
  <main id="results">{empty_message}{''.join(groups_html)}</main>
  <nav id="pagination" class="pagination" hidden></nav>
  <script>
    const root=document.getElementById('results');
    const groups=Array.from(root.querySelectorAll('.result-group'));
    const filter=document.getElementById('filter'); const video=document.getElementById('video');
    const sort=document.getElementById('sort'); const pageSize=document.getElementById('page-size');
    const pagination=document.getElementById('pagination');
    let currentPage=1;
    function pageButton(label,page,opts) {{
      const button=document.createElement('button');
      button.type='button'; button.textContent=label;
      if(opts&&opts.active)button.classList.add('active');
      if(opts&&opts.disabled)button.disabled=true;
      else button.addEventListener('click',()=>{{
        currentPage=page; update(); window.scrollTo({{top:0,behavior:'smooth'}});
      }});
      return button;
    }}
    function renderPagination(pages,total) {{
      pagination.replaceChildren();
      pagination.hidden=pages<=1;
      if(pages<=1)return;
      pagination.appendChild(pageButton('‹',currentPage-1,{{disabled:currentPage===1}}));
      const shown=[...new Set([1,pages,currentPage-1,currentPage,currentPage+1])]
        .filter(p=>p>=1&&p<=pages).sort((a,b)=>a-b);
      let previous=0;
      shown.forEach(p=>{{
        if(p-previous>1){{const gap=document.createElement('span');gap.className='gap';gap.textContent='…';pagination.appendChild(gap);}}
        pagination.appendChild(pageButton(String(p),p,{{active:p===currentPage}}));
        previous=p;
      }});
      pagination.appendChild(pageButton('›',currentPage+1,{{disabled:currentPage===pages}}));
      const info=document.createElement('span');
      info.className='page-info';
      info.textContent=`trang ${{currentPage}}/${{pages}} · ${{total}} kết quả`;
      pagination.appendChild(info);
    }}
    function update() {{
      const q=filter.value.trim().toLocaleLowerCase(); const v=video.value;
      const matched=groups.filter(g=>(!q||g.dataset.search.includes(q))&&(!v||g.dataset.video.split(' ').includes(v)));
      matched.sort((a,b)=>sort.value==='score' ? Number(b.dataset.score)-Number(a.dataset.score) : sort.value==='video' ? a.dataset.video.localeCompare(b.dataset.video) : Number(a.dataset.rank)-Number(b.dataset.rank));
      const size=pageSize.value==='all' ? Math.max(matched.length,1) : Number(pageSize.value);
      const pages=Math.max(1,Math.ceil(matched.length/size));
      if(currentPage>pages)currentPage=pages;
      if(currentPage<1)currentPage=1;
      groups.forEach(g=>g.classList.add('hidden'));
      matched.forEach((g,index)=>{{
        root.appendChild(g);
        g.classList.toggle('hidden',Math.floor(index/size)+1!==currentPage);
      }});
      renderPagination(pages,matched.length);
    }}
    function resetToFirstPage() {{ currentPage=1; update(); }}
    filter.addEventListener('input',resetToFirstPage); video.addEventListener('change',resetToFirstPage);
    sort.addEventListener('change',resetToFirstPage); pageSize.addEventListener('change',resetToFirstPage);
    update();
    document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{{
      try {{ await navigator.clipboard.writeText(button.dataset.copy); button.textContent='đã copy'; }}
      catch (_) {{ button.textContent=button.dataset.copy; }}
    }}));
  </script>
</body>
</html>
"""
