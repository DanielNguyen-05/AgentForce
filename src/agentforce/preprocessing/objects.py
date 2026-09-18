"""Normalize organizer Open Images detections into searchable records."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unicodedata
from typing import Iterable, Mapping

from agentforce.data.schemas import BoundingBox, ObjectDetection, ObjectFrame, write_jsonl


def normalize_label(label: str) -> str:
    return " ".join(unicodedata.normalize("NFC", label).strip().split())


def intersection_over_union(left: BoundingBox, right: BoundingBox) -> float:
    x_min = max(left.x_min, right.x_min)
    y_min = max(left.y_min, right.y_min)
    x_max = min(left.x_max, right.x_max)
    y_max = min(left.y_max, right.y_max)
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    union = left.area + right.area - intersection
    return intersection / union if union > 0 else 0.0


def classwise_nms(
    detections: Iterable[ObjectDetection], *, iou_threshold: float = 0.5
) -> list[ObjectDetection]:
    """Class-wise non-maximum suppression for meaningful object counts."""

    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    kept: list[ObjectDetection] = []
    by_class: dict[str, list[ObjectDetection]] = {}
    for detection in detections:
        by_class.setdefault(detection.class_mid or detection.label.casefold(), []).append(detection)
    for class_detections in by_class.values():
        class_kept: list[ObjectDetection] = []
        for candidate in sorted(class_detections, key=lambda item: item.score, reverse=True):
            if all(
                intersection_over_union(candidate.bbox, accepted.bbox) <= iou_threshold
                for accepted in class_kept
            ):
                class_kept.append(candidate)
        kept.extend(class_kept)
    return sorted(kept, key=lambda item: item.score, reverse=True)


def load_object_frame(
    path: str | Path,
    keyframe_uid: str,
    *,
    min_score: float = 0.2,
    class_thresholds: Mapping[str, float] | None = None,
    label_aliases_vi: Mapping[str, str] | None = None,
    iou_threshold: float | None = 0.5,
    max_detections: int | None = 100,
) -> ObjectFrame:
    """Load one organizer JSON, filter confidence and suppress duplicates.

    Per-class thresholds may be keyed by Open Images MID or English entity.
    Organizer boxes arrive as ``y_min, x_min, y_max, x_max`` and are converted
    immediately to the canonical x/y order.
    """

    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be in [0, 1]")
    if max_detections is not None and max_detections < 0:
        raise ValueError("max_detections cannot be negative")
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    scores = payload.get("detection_scores", [])
    mids = payload.get("detection_class_names", [])
    labels = payload.get("detection_class_entities", [])
    boxes = payload.get("detection_boxes", [])
    lengths = {len(scores), len(mids), len(labels), len(boxes)}
    if len(lengths) != 1:
        raise ValueError(
            f"Parallel detection arrays differ in length for {source}: "
            f"scores={len(scores)}, mids={len(mids)}, labels={len(labels)}, boxes={len(boxes)}"
        )

    thresholds = class_thresholds or {}
    aliases = label_aliases_vi or {}
    detections: list[ObjectDetection] = []
    for index, (raw_score, raw_mid, raw_label, raw_box) in enumerate(
        zip(scores, mids, labels, boxes, strict=True)
    ):
        try:
            score = float(raw_score)
            y_min, x_min, y_max, x_max = (float(value) for value in raw_box)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid detection {index} in {source}") from exc
        mid = str(raw_mid).strip()
        raw_entity = normalize_label(str(raw_label)) or mid
        label = raw_entity.casefold()
        threshold = float(
            thresholds.get(mid, thresholds.get(raw_entity, thresholds.get(label, min_score)))
        )
        if score < threshold:
            continue
        bbox = BoundingBox(
            x_min=max(0.0, min(1.0, x_min)),
            y_min=max(0.0, min(1.0, y_min)),
            x_max=max(0.0, min(1.0, x_max)),
            y_max=max(0.0, min(1.0, y_max)),
            normalized=True,
        )
        alias = aliases.get(mid, aliases.get(raw_entity, aliases.get(label)))
        detections.append(
            ObjectDetection(
                class_mid=mid,
                label=label,
                label_vi=normalize_label(alias) if alias else None,
                score=score,
                bbox=bbox,
            )
        )

    if iou_threshold is not None:
        detections = classwise_nms(detections, iou_threshold=iou_threshold)
    if max_detections is not None:
        detections = detections[:max_detections]
    counts = Counter(detection.label for detection in detections)
    aliases_by_label = {
        detection.label: detection.label_vi
        for detection in detections
        if detection.label_vi is not None
    }
    phrases: list[str] = []
    for label, count in sorted(counts.items()):
        alias = aliases_by_label.get(label)
        display = f"{label}/{alias}" if alias else label
        phrases.append(f"{count} {display}")
    return ObjectFrame(
        keyframe_uid=keyframe_uid,
        detections=detections,
        counts=dict(sorted(counts.items())),
        searchable_text=", ".join(phrases),
        source_path=str(source),
    )


def normalize_object_files(
    items: Iterable[tuple[str, str | Path]],
    output_path: str | Path,
    **kwargs: object,
) -> None:
    """Normalize ``(keyframe_uid, json_path)`` pairs to one JSONL shard."""

    frames = (load_object_frame(path, uid, **kwargs) for uid, path in items)
    write_jsonl(frames, output_path)
