from __future__ import annotations

from agentforce.data.schemas import KeyframeRecord, TranscriptSegment, TranscriptWord
from agentforce.preprocessing.windows import (
    WindowConfig,
    align_transcript_to_windows,
    build_temporal_windows,
)


def _keyframe(number: int, timestamp: float) -> KeyframeRecord:
    return KeyframeRecord(
        keyframe_uid=f"L01_V001_K{number:06d}",
        video_id="L01_V001",
        keyframe_number=number,
        frame_idx=round(timestamp * 25),
        pts_time=timestamp,
        fps=25,
        visual_embedding_row=number - 1,
    )


def test_overlapping_windows_retain_boundary_spanning_transcript() -> None:
    keyframes = [_keyframe(1, 1), _keyframe(2, 4.5), _keyframe(3, 6)]
    windows = build_temporal_windows(
        "L01_V001",
        keyframes,
        duration_seconds=10,
        config=WindowConfig(length_seconds=5, stride_seconds=5),
    )
    segment = TranscriptSegment(segment_id=7, start=4, end=6, text="qua ranh giới")
    align_transcript_to_windows(windows, [segment], min_segment_overlap=0.2)
    assert [window.asr_segment_ids for window in windows] == [[7], [7]]
    assert windows[0].end_frame == 124
    assert windows[1].start_frame == 125


def test_rounded_media_duration_is_extended_to_include_last_keyframe() -> None:
    keyframes = [_keyframe(1, 0), _keyframe(2, 10.02)]
    windows = build_temporal_windows(
        "L01_V001",
        keyframes,
        duration_seconds=10,
        config=WindowConfig(length_seconds=5, stride_seconds=5),
    )
    assert keyframes[-1].keyframe_uid in windows[-1].keyframe_uids
    assert windows[-1].end_time >= 10.02


def test_word_timestamps_are_preferred_over_long_segment_text() -> None:
    keyframes = [_keyframe(1, 1), _keyframe(2, 6)]
    windows = build_temporal_windows(
        "L01_V001",
        keyframes,
        duration_seconds=10,
        config=WindowConfig(length_seconds=5, stride_seconds=5),
    )
    segment = TranscriptSegment(
        segment_id=9,
        start=0,
        end=10,
        text="không được sao chép cả đoạn dài",
        words=[
            TranscriptWord("đầu", 1.0, 1.4, 0.9),
            TranscriptWord("giữa", 4.8, 5.2, 0.9),
            TranscriptWord("cuối", 8.0, 8.4, 0.9),
        ],
    )

    align_transcript_to_windows(windows, [segment], min_segment_overlap=0.9)

    assert windows[0].asr_text == "đầu"
    assert windows[1].asr_text == "giữa cuối"
    assert [window.asr_segment_ids for window in windows] == [[9], [9]]
