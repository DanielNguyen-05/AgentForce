"""Typed inputs for ranking evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class TaskType(str, Enum):
    KIS = "kis"
    QA = "qa"
    TRAKE = "trake"

    @classmethod
    def parse(cls, value: "TaskType | str") -> "TaskType":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"Unsupported task type: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class FrameInterval:
    """Inclusive frame interval."""

    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        for name, value in (("start_frame", self.start_frame), ("end_frame", self.end_frame)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be >= start_frame")

    def contains(self, frame_idx: int) -> bool:
        return self.start_frame <= frame_idx <= self.end_frame


@dataclass(frozen=True, slots=True)
class VideoFrameInterval:
    video_id: str
    interval: FrameInterval

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string")
        object.__setattr__(self, "video_id", self.video_id.strip())
        if not isinstance(self.interval, FrameInterval):
            raise TypeError("interval must be a FrameInterval")

    @classmethod
    def from_bounds(cls, video_id: str, start_frame: int, end_frame: int) -> "VideoFrameInterval":
        return cls(video_id=video_id, interval=FrameInterval(start_frame, end_frame))


@dataclass(frozen=True, slots=True)
class KISGroundTruth:
    relevant_intervals: tuple[VideoFrameInterval, ...] | Sequence[VideoFrameInterval]

    def __post_init__(self) -> None:
        intervals = tuple(self.relevant_intervals)
        if not intervals:
            raise ValueError("KIS ground truth requires at least one relevant interval")
        if not all(isinstance(item, VideoFrameInterval) for item in intervals):
            raise TypeError("relevant_intervals must contain VideoFrameInterval values")
        object.__setattr__(self, "relevant_intervals", intervals)

    @property
    def task_type(self) -> TaskType:
        return TaskType.KIS


@dataclass(frozen=True, slots=True)
class QAGroundTruth:
    relevant_intervals: tuple[VideoFrameInterval, ...] | Sequence[VideoFrameInterval]
    accepted_answers: tuple[str, ...] | Sequence[str]

    def __post_init__(self) -> None:
        intervals = tuple(self.relevant_intervals)
        answers = tuple(answer.strip() for answer in self.accepted_answers if isinstance(answer, str) and answer.strip())
        if not intervals:
            raise ValueError("Q&A ground truth requires at least one relevant interval")
        if not all(isinstance(item, VideoFrameInterval) for item in intervals):
            raise TypeError("relevant_intervals must contain VideoFrameInterval values")
        if not answers:
            raise ValueError("Q&A ground truth requires at least one accepted answer")
        object.__setattr__(self, "relevant_intervals", intervals)
        object.__setattr__(self, "accepted_answers", answers)

    @property
    def task_type(self) -> TaskType:
        return TaskType.QA


@dataclass(frozen=True, slots=True)
class TRAKEEventGroundTruth:
    """One event may accept multiple semantic frame intervals."""

    intervals: tuple[FrameInterval, ...] | Sequence[FrameInterval]

    def __post_init__(self) -> None:
        intervals = tuple(self.intervals)
        if not intervals:
            raise ValueError("A TRAKE event requires at least one interval")
        if not all(isinstance(item, FrameInterval) for item in intervals):
            raise TypeError("intervals must contain FrameInterval values")
        object.__setattr__(self, "intervals", intervals)

    def contains(self, frame_idx: int) -> bool:
        return any(interval.contains(frame_idx) for interval in self.intervals)


@dataclass(frozen=True, slots=True)
class TRAKEGroundTruth:
    video_id: str
    events: tuple[TRAKEEventGroundTruth, ...] | Sequence[TRAKEEventGroundTruth]

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string")
        events = tuple(self.events)
        if not events:
            raise ValueError("TRAKE ground truth requires at least one event")
        if not all(isinstance(item, TRAKEEventGroundTruth) for item in events):
            raise TypeError("events must contain TRAKEEventGroundTruth values")
        object.__setattr__(self, "video_id", self.video_id.strip())
        object.__setattr__(self, "events", events)

    @property
    def task_type(self) -> TaskType:
        return TaskType.TRAKE


GroundTruth = KISGroundTruth | QAGroundTruth | TRAKEGroundTruth


@dataclass(frozen=True, slots=True)
class RankedPrediction:
    """One ranked submission candidate; list order defines rank."""

    video_id: str
    frame_ids: tuple[int, ...] | Sequence[int]
    answer: str | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string")
        frames = tuple(self.frame_ids)
        if not frames:
            raise ValueError("A prediction requires at least one frame ID")
        for frame_idx in frames:
            if isinstance(frame_idx, bool) or not isinstance(frame_idx, int) or frame_idx < 0:
                raise ValueError("frame_ids must contain non-negative integers")
        if self.answer is not None and not isinstance(self.answer, str):
            raise ValueError("answer must be a string or null")
        if self.score is not None:
            object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "video_id", self.video_id.strip())
        object.__setattr__(self, "frame_ids", frames)

    @classmethod
    def kis(cls, video_id: str, frame_idx: int, *, score: float | None = None) -> "RankedPrediction":
        return cls(video_id=video_id, frame_ids=(frame_idx,), score=score)

    @classmethod
    def qa(
        cls,
        video_id: str,
        frame_idx: int,
        answer: str,
        *,
        score: float | None = None,
    ) -> "RankedPrediction":
        return cls(video_id=video_id, frame_ids=(frame_idx,), answer=answer, score=score)

    @classmethod
    def trake(
        cls,
        video_id: str,
        frame_ids: Sequence[int],
        *,
        score: float | None = None,
    ) -> "RankedPrediction":
        return cls(video_id=video_id, frame_ids=tuple(frame_ids), score=score)
