"""Build overlapping retrieval windows and align timestamped sidecars."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from agentforce.data.schemas import (
    KeyframeRecord,
    TemporalWindow,
    TranscriptSegment,
    TranscriptWord,
)


@dataclass(frozen=True, slots=True)
class WindowConfig:
    length_seconds: float = 10.0
    stride_seconds: float = 5.0
    include_empty: bool = False

    def __post_init__(self) -> None:
        if self.length_seconds <= 0 or self.stride_seconds <= 0:
            raise ValueError("Window length and stride must be positive")
        if self.stride_seconds > self.length_seconds:
            raise ValueError("stride_seconds cannot exceed length_seconds (it would leave gaps)")


def build_temporal_windows(
    video_id: str,
    keyframes: Sequence[KeyframeRecord],
    *,
    duration_seconds: float | None = None,
    config: WindowConfig | None = None,
) -> list[TemporalWindow]:
    """Create overlapping windows while retaining exact frame boundaries."""

    settings = config or WindowConfig()
    if any(record.video_id != video_id for record in keyframes):
        raise ValueError("All keyframes must match video_id")
    fps = keyframes[0].fps if keyframes else 25.0
    if duration_seconds is None:
        duration_seconds = (keyframes[-1].pts_time + 1.0 / fps) if keyframes else 0.0
    if duration_seconds < 0:
        raise ValueError("duration_seconds cannot be negative")
    # Organizer/YouTube metadata stores a rounded integer length for some files.
    # Never drop the final canonical keyframe just because that value rounded down.
    if keyframes:
        duration_seconds = max(duration_seconds, keyframes[-1].pts_time + 1.0 / fps)
    if duration_seconds == 0:
        return []

    windows: list[TemporalWindow] = []
    start = 0.0
    candidate_number = 0
    while start < duration_seconds:
        end = min(start + settings.length_seconds, duration_seconds)
        is_last = end >= duration_seconds
        members = [
            record
            for record in keyframes
            if start <= record.pts_time < end or (is_last and record.pts_time == end)
        ]
        if members or settings.include_empty:
            representative = (
                min(members, key=lambda record: abs(record.pts_time - (start + end) / 2.0))
                if members
                else None
            )
            windows.append(
                TemporalWindow(
                    window_id=f"{video_id}_W{candidate_number:06d}",
                    video_id=video_id,
                    window_number=candidate_number,
                    start_time=start,
                    end_time=end,
                    start_frame=max(0, math.floor(start * fps)),
                    end_frame=max(0, math.ceil(end * fps) - 1),
                    representative_keyframe_uid=(
                        representative.keyframe_uid if representative is not None else None
                    ),
                    frame_idx=(representative.frame_idx if representative is not None else None),
                    pts_time=(representative.pts_time if representative is not None else None),
                    fps=fps,
                    keyframe_uids=[record.keyframe_uid for record in members],
                )
            )
        candidate_number += 1
        start = candidate_number * settings.stride_seconds
    return windows


def interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def align_transcript_to_windows(
    windows: Sequence[TemporalWindow],
    segments: Sequence[TranscriptSegment],
    *,
    min_segment_overlap: float = 0.2,
) -> Sequence[TemporalWindow]:
    """Attach timestamped ASR text to retrieval windows.

    Word timestamps are preferred because Whisper/PhoWhisper segments are often
    around 30 seconds long, substantially wider than a retrieval window.  A
    word is assigned by its timestamp midpoint, which avoids copying the full
    long segment into every intersecting window.  Segment overlap remains the
    fallback for transcripts that do not contain usable word timestamps.
    """

    if not 0 <= min_segment_overlap <= 1:
        raise ValueError("min_segment_overlap must be in [0, 1]")
    for window in windows:
        selected_ids: list[int] = []
        text_parts: list[str] = []
        for segment in segments:
            if segment.end < segment.start:
                raise ValueError(f"ASR segment {segment.segment_id} has end < start")
            for word in segment.words:
                if word.end < word.start:
                    raise ValueError(
                        f"ASR word in segment {segment.segment_id} has end < start"
                    )
            usable_words = [word for word in segment.words if word.text.strip()]
            if usable_words:
                selected_words: list[TranscriptWord] = []
                for word in usable_words:
                    midpoint = (word.start + word.end) / 2.0
                    if window.start_time <= midpoint < window.end_time:
                        selected_words.append(word)
                if selected_words:
                    selected_ids.append(segment.segment_id)
                    text_parts.extend(word.text.strip() for word in selected_words)
                continue

            duration = max(1e-9, segment.end - segment.start)
            overlap = interval_overlap(
                window.start_time, window.end_time, segment.start, segment.end
            )
            if overlap > 0 and overlap / duration >= min_segment_overlap:
                selected_ids.append(segment.segment_id)
                if segment.text.strip():
                    text_parts.append(segment.text.strip())
        window.asr_segment_ids = list(dict.fromkeys(selected_ids))
        window.asr_text = " ".join(text_parts)
    return windows


def attach_keyframe_text(
    windows: Sequence[TemporalWindow],
    text_by_keyframe_uid: dict[str, str],
    *,
    target_field: str,
) -> Sequence[TemporalWindow]:
    """Aggregate keyframe OCR text into existing windows without duplication."""

    if target_field != "ocr_text":
        raise ValueError("target_field must be 'ocr_text'")
    for window in windows:
        values = [
            text_by_keyframe_uid[uid].strip()
            for uid in window.keyframe_uids
            if uid in text_by_keyframe_uid and text_by_keyframe_uid[uid].strip()
        ]
        deduplicated = list(dict.fromkeys(values))
        setattr(window, target_field, "\n".join(deduplicated))
    return windows


def attach_object_labels(
    windows: Sequence[TemporalWindow], labels_by_keyframe_uid: dict[str, Iterable[str]]
) -> Sequence[TemporalWindow]:
    for window in windows:
        labels = {
            label
            for uid in window.keyframe_uids
            for label in labels_by_keyframe_uid.get(uid, [])
            if label
        }
        window.object_labels = sorted(labels)
    return windows
