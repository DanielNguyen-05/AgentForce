"""Normalize retrieval/task results and render a safe standalone keyframe gallery."""

from __future__ import annotations

import base64
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape
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


def _record_card(
    record: GalleryRecord,
    *,
    image_src: str | None,
    image_error: str | None,
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
    media_badge_class = "warn" if record.media.resolution != "exact_keyframe" else "ok"
    badges = [
        _badge(media_label, media_badge_class),
        *(_badge(name, f"mod-{name}") for name in sorted(record.modality_scores)),
    ]
    image_html = (
        f'<a class="image-link" href="{image_src}" target="_blank" rel="noopener">'
        f'<img loading="lazy" src="{image_src}" alt="{escape(record.record_id, quote=True)}"></a>'
        if image_src
        else f'<div class="missing">{escape(image_error or "Không có ảnh")}</div>'
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
        objects = "".join(_badge(label, "object") for label in record.object_labels)
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
        <div class="card-head"><strong>{title}</strong><div>{''.join(badges)}</div></div>
        {image_html}
        <div class="meta">
          <span><b>Video</b> {escape(record.video_id)}</span>
          <span><b>Frame</b> {requested}{escape(display_note)}</span>
          <span><b>Time</b> {_format_timestamp(record.timestamp)}</span>
          <span><b>Score</b> {_score(record.score)}</span>
        </div>
        <div class="actions">
          <button type="button" data-copy="{escape(f'{record.video_id},{requested}', quote=True)}">copy video,frame</button>
          {original_link}
        </div>
        <details class="context"><summary>Evidence chi tiết</summary>{''.join(context_parts)}</details>
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
                _record_card(record, image_src=image_src, image_error=image_error)
            )
        group_score = records[0].group_score
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
        groups_html.append(
            f"""
            <section class="result-group" data-rank="{rank}" data-score="{group_score or 0.0}"
              data-video="{escape(' '.join(videos), quote=True)}"
              data-search="{escape(searchable, quote=True)}">
              <h2><span>Rank #{rank}</span><span>score {_score(group_score)}</span></h2>
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
        else '<div class="empty">Không có frame nào trong result JSON.</div>'
    )
    query_id = document.query_id or "—"
    title = f"{document.task_type.upper()} · {query_id}"
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; object-src 'none'; base-uri 'none'">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#f4f6fa; --panel:#fff; --text:#172033;
      --muted:#667085; --line:#d9deea; --accent:#3157d5; --good:#087443; --warn:#a15c00; }}
    @media (prefers-color-scheme:dark) {{ :root {{ --bg:#0d111b; --panel:#161d2b;
      --text:#ecf1ff; --muted:#a7b0c3; --line:#303a4e; --accent:#8ea8ff;
      --good:#69d4a0; --warn:#ffbd66; }} }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text);
      font-family:ui-sans-serif,system-ui,-apple-system,sans-serif; line-height:1.45; }}
    header {{ position:sticky; top:0; z-index:10; padding:18px 24px; background:color-mix(in srgb,var(--panel) 94%,transparent);
      border-bottom:1px solid var(--line); backdrop-filter:blur(10px); }}
    h1 {{ margin:0 0 7px; font-size:20px; }} .query {{ margin:0; font-size:16px; }}
    .summary {{ color:var(--muted); font-size:13px; margin-top:7px; }}
    .controls {{ display:grid; grid-template-columns:minmax(220px,1fr) 180px 180px;
      gap:10px; margin-top:14px; }} input,select,button {{ border:1px solid var(--line);
      background:var(--panel); color:var(--text); border-radius:8px; padding:9px 11px; }}
    main {{ padding:20px 24px 50px; max-width:1800px; margin:auto; }}
    .result-group {{ margin-bottom:25px; }} .result-group>h2 {{ display:flex;
      justify-content:space-between; margin:0 0 9px; font-size:15px; color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat({columns},minmax(0,1fr)); gap:14px; }}
    .card {{ min-width:0; background:var(--panel); border:1px solid var(--line);
      border-radius:12px; overflow:hidden; box-shadow:0 3px 14px #0001; }}
    .card-head {{ padding:10px 12px; display:flex; justify-content:space-between;
      gap:8px; align-items:flex-start; }} .card-head strong {{ overflow-wrap:anywhere; }}
    .image-link {{ display:block; background:#05070a; }} img {{ display:block; width:100%;
      aspect-ratio:16/9; object-fit:contain; }} .missing {{ aspect-ratio:16/9; display:grid;
      place-items:center; padding:20px; color:var(--warn); background:#0002; text-align:center; }}
    .meta {{ display:grid; grid-template-columns:1fr 1fr; gap:5px 12px; padding:10px 12px;
      font-size:13px; }} .actions {{ display:flex; justify-content:space-between; align-items:center;
      padding:0 12px 10px; }} .actions button {{ cursor:pointer; padding:5px 8px; font-size:12px; }}
    .path {{ color:var(--accent); font-size:12px; }} .badge {{ display:inline-block;
      padding:2px 6px; margin:0 0 3px 4px; border:1px solid var(--line); border-radius:999px;
      color:var(--muted); font-size:10px; }} .badge.ok {{ color:var(--good); }}
    .badge.warn {{ color:var(--warn); }} .badge.object {{ margin-left:0; margin-right:4px; }}
    .mod-visual {{ color:#8255cf; }} .mod-asr {{ color:#087443; }}
    .mod-ocr {{ color:#b15c00; }} .mod-objects {{ color:#0f6fa8; }}
    details.context {{ border-top:1px solid var(--line); padding:9px 12px 12px; }}
    details summary {{ cursor:pointer; color:var(--accent); }} section h4 {{ margin:12px 0 4px; }}
    section p {{ margin:0; white-space:pre-wrap; overflow-wrap:anywhere; }} .answer {{ font-weight:700; }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; max-height:280px; overflow:auto;
      padding:9px; border-radius:7px; background:#0001; font-size:11px; }}
    .scores {{ width:100%; border-collapse:collapse; font-size:12px; }}
    .scores td {{ padding:2px 4px; border-bottom:1px solid var(--line); }} .muted {{ color:var(--muted); }}
    .empty {{ padding:40px; text-align:center; color:var(--muted); }} .hidden {{ display:none; }}
    @media(max-width:1000px) {{ .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media(max-width:650px) {{ header,main {{ padding-left:12px; padding-right:12px; }}
      .controls,.cards {{ grid-template-columns:1fr; }} header {{ position:static; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <p class="query">{escape(document.query_text)}</p>
    <div class="summary">{len(grouped)} ranked results · {len(document.records)} frames · nguồn {escape(document.source_label)} · tạo lúc {escape(document.generated_at)}</div>
    <div class="controls">
      <input id="filter" type="search" placeholder="lọc ASR, OCR, object, candidate...">
      <select id="video"><option value="">tất cả video</option>{video_options}</select>
      <select id="sort"><option value="rank">xếp theo rank</option><option value="score">xếp theo score</option><option value="video">xếp theo video</option></select>
    </div>
  </header>
  <main id="results">{empty_message}{''.join(groups_html)}</main>
  <script>
    const root=document.getElementById('results');
    const groups=Array.from(root.querySelectorAll('.result-group'));
    const filter=document.getElementById('filter'); const video=document.getElementById('video');
    const sort=document.getElementById('sort');
    function update() {{
      const q=filter.value.trim().toLocaleLowerCase(); const v=video.value;
      groups.forEach(g=>g.classList.toggle('hidden',!((!q||g.dataset.search.includes(q))&&(!v||g.dataset.video.split(' ').includes(v)))));
      const ordered=[...groups].sort((a,b)=>sort.value==='score' ? Number(b.dataset.score)-Number(a.dataset.score) : sort.value==='video' ? a.dataset.video.localeCompare(b.dataset.video) : Number(a.dataset.rank)-Number(b.dataset.rank));
      ordered.forEach(g=>root.appendChild(g));
    }}
    filter.addEventListener('input',update); video.addEventListener('change',update); sort.addEventListener('change',update);
    document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{{
      try {{ await navigator.clipboard.writeText(button.dataset.copy); button.textContent='đã copy'; }}
      catch (_) {{ button.textContent=button.dataset.copy; }}
    }}));
  </script>
</body>
</html>
"""
