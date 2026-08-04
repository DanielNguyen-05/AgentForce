"""Dynamic-programming alignment for ordered TRAKE events."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .refinement import FrameCandidate


@dataclass(frozen=True, slots=True)
class ViterbiPath:
    indices: tuple[int, ...]
    positions: tuple[float, ...]
    event_scores: tuple[float, ...]
    total_score: float


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    video_id: str
    candidates: tuple[FrameCandidate, ...]
    score: float

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(candidate.frame_idx for candidate in self.candidates)


def _validate_matrix(
    event_scores: ArrayLike, positions: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    scores = np.asarray(event_scores, dtype=np.float64)
    timeline = np.asarray(positions, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("event_scores must be a non-empty [events, positions] matrix")
    if timeline.ndim != 1 or timeline.shape[0] != scores.shape[1]:
        raise ValueError("positions must be a vector matching the matrix width")
    if not np.isfinite(timeline).all() or np.any(np.diff(timeline) < 0):
        raise ValueError("positions must be finite and sorted in ascending order")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("scores may contain -inf for invalid states, but not NaN/+inf")
    return scores, timeline


def _can_precede(
    previous: float,
    current: float,
    *,
    strict: bool,
    min_gap: float,
    max_gap: float | None,
) -> bool:
    gap = current - previous
    if strict and gap <= 0:
        return False
    if gap < min_gap:
        return False
    return max_gap is None or gap <= max_gap


def ordered_viterbi(
    event_scores: ArrayLike,
    positions: ArrayLike,
    *,
    strict: bool = True,
    min_gap: float = 0.0,
    max_gap: float | None = None,
    transition_penalty: Callable[[float], float] | None = None,
) -> ViterbiPath | None:
    """Find the maximum-scoring ordered event path on a shared timeline.

    With no custom transition penalty this uses a monotonic deque and runs in
    ``O(events * positions)``. Supplying a penalty intentionally selects the more
    general ``O(events * positions**2)`` recurrence, suitable for short candidate
    lists and experiments with target inter-event gaps.
    """

    if min_gap < 0 or (max_gap is not None and max_gap < min_gap):
        raise ValueError("gap constraints are invalid")
    scores, timeline = _validate_matrix(event_scores, positions)
    event_count, position_count = scores.shape
    previous_scores = scores[0].copy()
    backpointers = np.full((event_count, position_count), -1, dtype=np.int64)

    for event_index in range(1, event_count):
        current_scores = np.full(position_count, -np.inf, dtype=np.float64)
        if transition_penalty is not None:
            for current_index in range(position_count):
                if not np.isfinite(scores[event_index, current_index]):
                    continue
                best_score = -np.inf
                best_previous = -1
                for previous_index in range(position_count):
                    if not np.isfinite(previous_scores[previous_index]):
                        continue
                    if not _can_precede(
                        timeline[previous_index],
                        timeline[current_index],
                        strict=strict,
                        min_gap=min_gap,
                        max_gap=max_gap,
                    ):
                        continue
                    gap = float(timeline[current_index] - timeline[previous_index])
                    value = previous_scores[previous_index] - transition_penalty(gap)
                    if value > best_score:
                        best_score, best_previous = value, previous_index
                if best_previous >= 0:
                    current_scores[current_index] = (
                        best_score + scores[event_index, current_index]
                    )
                    backpointers[event_index, current_index] = best_previous
        else:
            candidates: deque[int] = deque()
            add_index = 0
            for current_index in range(position_count):
                upper = timeline[current_index] - min_gap
                while add_index < position_count:
                    allowed = timeline[add_index] < upper if strict and min_gap == 0 else timeline[add_index] <= upper
                    if not allowed:
                        break
                    if np.isfinite(previous_scores[add_index]):
                        while (
                            candidates
                            and previous_scores[candidates[-1]] < previous_scores[add_index]
                        ):
                            candidates.pop()
                        candidates.append(add_index)
                    add_index += 1
                if max_gap is not None:
                    lower = timeline[current_index] - max_gap
                    while candidates and timeline[candidates[0]] < lower:
                        candidates.popleft()
                if candidates and np.isfinite(scores[event_index, current_index]):
                    predecessor = candidates[0]
                    current_scores[current_index] = (
                        previous_scores[predecessor] + scores[event_index, current_index]
                    )
                    backpointers[event_index, current_index] = predecessor
        previous_scores = current_scores

    if not np.isfinite(previous_scores).any():
        return None
    final_index = int(np.argmax(previous_scores))
    path = [final_index]
    for event_index in range(event_count - 1, 0, -1):
        final_index = int(backpointers[event_index, final_index])
        if final_index < 0:
            return None
        path.append(final_index)
    path.reverse()
    return ViterbiPath(
        indices=tuple(path),
        positions=tuple(float(timeline[index]) for index in path),
        event_scores=tuple(float(scores[event, index]) for event, index in enumerate(path)),
        total_score=float(sum(scores[event, index] for event, index in enumerate(path))),
    )


def _candidate_position(candidate: FrameCandidate) -> float:
    return candidate.timestamp


def align_ordered_candidates(
    candidates_by_event: Sequence[Sequence[FrameCandidate]],
    *,
    video_id: str | None = None,
    strict: bool = True,
    min_gap: float = 0.0,
    max_gap: float | None = None,
    transition_penalty: Callable[[float], float] | None = None,
) -> AlignmentResult | None:
    """Align event-specific candidate lists, which may have different timelines."""

    if not candidates_by_event or any(not event for event in candidates_by_event):
        return None
    inferred_video = video_id or candidates_by_event[0][0].video_id
    filtered: list[list[FrameCandidate]] = []
    for event_candidates in candidates_by_event:
        matching = [item for item in event_candidates if item.video_id == inferred_video]
        matching.sort(key=lambda item: (_candidate_position(item), -item.score, item.frame_idx))
        if not matching:
            return None
        filtered.append(matching)

    previous_scores = np.asarray([item.score for item in filtered[0]], dtype=np.float64)
    backpointers: list[NDArray[np.int64]] = []
    for event_index in range(1, len(filtered)):
        previous = filtered[event_index - 1]
        current = filtered[event_index]
        current_scores = np.full(len(current), -np.inf, dtype=np.float64)
        pointers = np.full(len(current), -1, dtype=np.int64)
        for current_index, current_candidate in enumerate(current):
            for previous_index, previous_candidate in enumerate(previous):
                if not np.isfinite(previous_scores[previous_index]):
                    continue
                if not _can_precede(
                    _candidate_position(previous_candidate),
                    _candidate_position(current_candidate),
                    strict=strict,
                    min_gap=min_gap,
                    max_gap=max_gap,
                ):
                    continue
                gap = _candidate_position(current_candidate) - _candidate_position(
                    previous_candidate
                )
                penalty = transition_penalty(gap) if transition_penalty else 0.0
                value = previous_scores[previous_index] + current_candidate.score - penalty
                if value > current_scores[current_index]:
                    current_scores[current_index] = value
                    pointers[current_index] = previous_index
        backpointers.append(pointers)
        previous_scores = current_scores

    if not np.isfinite(previous_scores).any():
        return None
    index = int(np.argmax(previous_scores))
    total_score = float(previous_scores[index])
    selected = [filtered[-1][index]]
    for event_index in range(len(filtered) - 2, -1, -1):
        index = int(backpointers[event_index][index])
        if index < 0:
            return None
        selected.append(filtered[event_index][index])
    selected.reverse()
    return AlignmentResult(
        video_id=inferred_video, candidates=tuple(selected), score=total_score
    )

