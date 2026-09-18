"""Interfaces for exact-frame refinement without imposing a video decoder/model."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class FrameCandidate:
    """A coarse or refined frame on a video's canonical timeline."""

    candidate_id: str
    video_id: str
    frame_idx: int
    timestamp: float
    score: float
    source_candidate_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_idx < 0 or self.timestamp < 0:
            raise ValueError("frame index and timestamp must be non-negative")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """Decoder-neutral frame passed to a semantic scorer."""

    video_id: str
    frame_idx: int
    timestamp: float
    payload: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class DenseFrameSource(Protocol):
    """Implemented by OpenCV/PyAV adapters in the application layer."""

    def iter_frames(
        self,
        video_id: str,
        *,
        start_time: float,
        end_time: float,
        sample_fps: float,
    ) -> Iterable[DecodedFrame]: ...


class FrameScorer(Protocol):
    """Scores decoded frames against an event/query; larger is better."""

    def score(self, query: str, frames: Sequence[DecodedFrame]) -> Sequence[float]: ...


class DenseRefiner(Protocol):
    """Task solvers depend only on this small extension point."""

    def refine(
        self, query: str, candidates: Sequence[FrameCandidate]
    ) -> list[FrameCandidate]: ...


class DenseFrameRefiner:
    """Generic decode-and-score refinement around coarse timestamps.

    The class owns orchestration only. Video decoding and CLIP/VLM scoring remain
    replaceable adapters, which makes frame alignment testable without heavyweight
    optional dependencies.
    """

    def __init__(
        self,
        source: DenseFrameSource,
        scorer: FrameScorer,
        *,
        radius_seconds: float = 2.0,
        sample_fps: float = 25.0,
        top_per_candidate: int = 5,
        coarse_weight: float = 0.2,
        semantic_weight: float = 0.8,
    ) -> None:
        if radius_seconds <= 0 or sample_fps <= 0 or top_per_candidate <= 0:
            raise ValueError("radius, sample FPS, and top_per_candidate must be positive")
        if coarse_weight < 0 or semantic_weight < 0:
            raise ValueError("refinement weights must be non-negative")
        self._source = source
        self._scorer = scorer
        self._radius_seconds = radius_seconds
        self._sample_fps = sample_fps
        self._top_per_candidate = top_per_candidate
        self._coarse_weight = coarse_weight
        self._semantic_weight = semantic_weight

    def refine(
        self, query: str, candidates: Sequence[FrameCandidate]
    ) -> list[FrameCandidate]:
        refined: dict[tuple[str, int], FrameCandidate] = {}
        for coarse in candidates:
            frames = list(
                self._source.iter_frames(
                    coarse.video_id,
                    start_time=max(0.0, coarse.timestamp - self._radius_seconds),
                    end_time=coarse.timestamp + self._radius_seconds,
                    sample_fps=self._sample_fps,
                )
            )
            if not frames:
                refined[(coarse.video_id, coarse.frame_idx)] = coarse
                continue
            semantic_scores = list(self._scorer.score(query, frames))
            if len(semantic_scores) != len(frames):
                raise ValueError("frame scorer returned a different number of scores")
            local = sorted(
                zip(frames, semantic_scores, strict=True),
                key=lambda item: (-float(item[1]), item[0].frame_idx),
            )[: self._top_per_candidate]
            for frame, semantic_score in local:
                score = (
                    self._coarse_weight * coarse.score
                    + self._semantic_weight * float(semantic_score)
                )
                candidate = FrameCandidate(
                    candidate_id=f"{coarse.video_id}:{frame.frame_idx}",
                    video_id=coarse.video_id,
                    frame_idx=frame.frame_idx,
                    timestamp=frame.timestamp,
                    score=score,
                    source_candidate_id=coarse.candidate_id,
                    metadata={**coarse.metadata, **frame.metadata},
                )
                key = (candidate.video_id, candidate.frame_idx)
                previous = refined.get(key)
                if previous is None or candidate.score > previous.score:
                    refined[key] = candidate
        return sorted(
            refined.values(), key=lambda candidate: (-candidate.score, candidate.frame_idx)
        )


class NoOpDenseRefiner:
    """Useful default and explicit switch for coarse-only experiments."""

    def refine(
        self, query: str, candidates: Sequence[FrameCandidate]
    ) -> list[FrameCandidate]:
        return list(candidates)

