"""Ordered event retrieval and exact-frame alignment for TRAKE."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

from agentforce.retrieval.query import (
    HeuristicQueryParser,
    IdentityQueryExpander,
    QueryExpander,
    QueryParser,
)
from agentforce.retrieval.ranking import aggregate_by_video
from agentforce.retrieval.types import ParsedQuery, SearchHit, TaskType
from agentforce.temporal.alignment import AlignmentResult, align_ordered_candidates
from agentforce.temporal.refinement import DenseRefiner, FrameCandidate

from .common import TextSearcher, collect_text_results, fuse


@dataclass(frozen=True, slots=True)
class TRAKEConfig:
    per_source_k: int = 500
    fusion_pool_size: int = 1_000
    per_event_per_video: int = 30
    top_videos: int = 100
    top_k: int = 100
    fusion_method: str = "rrf"
    rank_constant: float = 60.0
    score_normalization: str = "minmax"
    modality_weights: Mapping[str, float] = field(default_factory=dict)
    global_video_weight: float = 0.2
    strict_order: bool = True
    min_event_gap_seconds: float = 0.0
    max_event_gap_seconds: float | None = None
    transition_penalty: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.per_source_k,
            self.fusion_pool_size,
            self.per_event_per_video,
            self.top_videos,
            self.top_k,
        ) <= 0:
            raise ValueError("retrieval sizes must be positive")
        if (
            self.global_video_weight < 0
            or self.min_event_gap_seconds < 0
            or self.transition_penalty < 0
        ):
            raise ValueError("weights and gaps must be non-negative")
        if (
            self.max_event_gap_seconds is not None
            and self.max_event_gap_seconds < self.min_event_gap_seconds
        ):
            raise ValueError("max event gap is smaller than min event gap")


@dataclass(frozen=True, slots=True)
class TRAKEAnswer:
    video_id: str
    frame_ids: tuple[int, ...]
    score: float
    event_candidate_ids: tuple[str, ...]
    event_timestamps: tuple[float, ...]
    evidence: Mapping[str, Any] = field(default_factory=dict)


class TRAKESolver:
    """Search every event independently, then enforce a monotonic event path."""

    def __init__(
        self,
        searchers: Mapping[str, TextSearcher] | None = None,
        *,
        parser: QueryParser | None = None,
        expander: QueryExpander | None = None,
        dense_refiner: DenseRefiner | None = None,
        config: TRAKEConfig | None = None,
    ) -> None:
        self._searchers = dict(searchers or {})
        self._parser = parser or HeuristicQueryParser()
        self._expander = expander or IdentityQueryExpander()
        self._dense_refiner = dense_refiner
        self._config = config or TRAKEConfig()

    def solve(self, query: str) -> list[TRAKEAnswer]:
        if not self._searchers:
            raise RuntimeError("TRAKESolver.solve requires at least one text searcher")
        parsed = self._parse(query)
        event_results: list[dict[str, Sequence[SearchHit]]] = []
        event_weights: list[dict[str, float]] = []
        for event in parsed.events:
            event_query = self._parser.parse(event.description, task_type=TaskType.KIS)
            results, weights = collect_text_results(
                event_query,
                searchers=self._searchers,
                expander=self._expander,
                per_source_k=self._config.per_source_k,
                modality_weights=self._config.modality_weights,
            )
            event_results.append(results)
            event_weights.append(weights)
        global_results, global_weights = collect_text_results(
            parsed,
            searchers=self._searchers,
            expander=IdentityQueryExpander(),
            per_source_k=self._config.per_source_k,
            modality_weights=self._config.modality_weights,
        )
        return self._solve_parsed(
            parsed,
            event_results,
            event_weights=event_weights,
            global_results=global_results,
            global_weights=global_weights,
        )

    def solve_from_results(
        self,
        query: str,
        event_results: Sequence[Mapping[str, Sequence[SearchHit]]],
        *,
        event_weights: Sequence[Mapping[str, float]] | None = None,
        global_results: Mapping[str, Sequence[SearchHit]] | None = None,
        global_weights: Mapping[str, float] | None = None,
    ) -> list[TRAKEAnswer]:
        parsed = self._parse(query)
        if len(event_results) != len(parsed.events):
            raise ValueError("event_results length must match parsed TRAKE events")
        weights = list(event_weights or ())
        if weights and len(weights) != len(event_results):
            raise ValueError("event_weights length must match event_results")
        if not weights:
            weights = [self._default_weights(results) for results in event_results]
        return self._solve_parsed(
            parsed,
            event_results,
            event_weights=weights,
            global_results=global_results,
            global_weights=global_weights or self._default_weights(global_results or {}),
        )

    def _parse(self, query: str) -> ParsedQuery:
        parsed = self._parser.parse(query, task_type=TaskType.TRAKE)
        if len(parsed.events) < 2:
            raise ValueError(
                "TRAKE query must contain at least two ordered events separated by "
                "'sau đó', 'rồi', 'tiếp theo', '->', or numbered lines"
            )
        return parsed

    def _default_weights(
        self, results: Mapping[str, Sequence[SearchHit]]
    ) -> dict[str, float]:
        return {
            source: float(self._config.modality_weights.get(source.split(":", 1)[0], 1.0))
            for source in results
        }

    def _solve_parsed(
        self,
        parsed: ParsedQuery,
        event_results: Sequence[Mapping[str, Sequence[SearchHit]]],
        *,
        event_weights: Sequence[Mapping[str, float]],
        global_results: Mapping[str, Sequence[SearchHit]] | None,
        global_weights: Mapping[str, float],
    ) -> list[TRAKEAnswer]:
        fused_events = [
            fuse(
                results,
                method=self._config.fusion_method,
                weights=weights,
                limit=self._config.fusion_pool_size,
                rank_constant=self._config.rank_constant,
                score_normalization=self._config.score_normalization,
            )
            for results, weights in zip(event_results, event_weights, strict=True)
        ]
        per_event_by_video: list[dict[str, list[FrameCandidate]]] = []
        for fused in fused_events:
            grouped: defaultdict[str, list[FrameCandidate]] = defaultdict(list)
            for hit in fused:
                candidate = self._as_frame_candidate(hit)
                if candidate is not None:
                    grouped[candidate.video_id].append(candidate)
            per_event_by_video.append(
                {
                    video_id: sorted(
                        candidates, key=lambda item: (-item.score, item.timestamp)
                    )[: self._config.per_event_per_video]
                    for video_id, candidates in grouped.items()
                }
            )
        if not per_event_by_video:
            return []
        eligible_videos = set(per_event_by_video[0])
        for grouped in per_event_by_video[1:]:
            eligible_videos.intersection_update(grouped)

        global_scores: dict[str, float] = {}
        if global_results:
            global_fused = fuse(
                global_results,
                method=self._config.fusion_method,
                weights=global_weights,
                limit=self._config.fusion_pool_size,
                rank_constant=self._config.rank_constant,
                score_normalization=self._config.score_normalization,
            )
            global_scores = {
                item.video_id: item.score
                for item in aggregate_by_video(global_fused)[: self._config.top_videos]
            }
            if global_scores:
                preferred = eligible_videos.intersection(global_scores)
                if preferred:
                    eligible_videos = preferred

        alignments: list[tuple[AlignmentResult, float]] = []
        for video_id in sorted(eligible_videos):
            event_candidates = [grouped[video_id] for grouped in per_event_by_video]
            # Multiple semantic events can share one coarse 10-second window.
            # Allow equal coarse timestamps when dense refinement can separate
            # them; enforce the official strict order on refined frames below.
            alignment = self._align(
                event_candidates,
                video_id=video_id,
                strict=False if self._dense_refiner else None,
            )
            if alignment is None:
                continue
            if self._dense_refiner:
                refined_events: list[list[FrameCandidate]] = []
                for event, coarse in zip(parsed.events, alignment.candidates, strict=True):
                    refined = self._dense_refiner.refine(event.description, [coarse])
                    refined_events.append(refined or [coarse])
                refined_alignment = self._align(
                    refined_events,
                    video_id=video_id,
                    strict=self._config.strict_order,
                )
                # Dense refinement is the final exact-frame stage.  Keeping a
                # non-strict coarse path after exact frames fail strict order
                # would emit an invalid TRAKE answer.
                if refined_alignment is None:
                    continue
                alignment = refined_alignment
            score = alignment.score / len(parsed.events)
            score += self._config.global_video_weight * global_scores.get(video_id, 0.0)
            alignments.append((alignment, score))
        alignments.sort(key=lambda item: (-item[1], item[0].video_id))

        return [
            TRAKEAnswer(
                video_id=alignment.video_id,
                frame_ids=alignment.frame_ids,
                score=score,
                event_candidate_ids=tuple(
                    candidate.candidate_id for candidate in alignment.candidates
                ),
                event_timestamps=tuple(
                    candidate.timestamp for candidate in alignment.candidates
                ),
                evidence={
                    "event_descriptions": [event.description for event in parsed.events],
                    "event_scores": [
                        candidate.score for candidate in alignment.candidates
                    ],
                    "global_video_score": global_scores.get(alignment.video_id, 0.0),
                },
            )
            for alignment, score in alignments[: self._config.top_k]
        ]

    def _align(
        self,
        event_candidates: Sequence[Sequence[FrameCandidate]],
        *,
        video_id: str,
        strict: bool | None = None,
    ) -> AlignmentResult | None:
        transition_penalty_fn = None
        if self._config.transition_penalty > 0:
            weight = self._config.transition_penalty

            def penalize_gap(gap: float) -> float:
                return weight * math.log1p(gap)

            transition_penalty_fn = penalize_gap
        return align_ordered_candidates(
            event_candidates,
            video_id=video_id,
            strict=self._config.strict_order if strict is None else strict,
            min_gap=self._config.min_event_gap_seconds,
            max_gap=self._config.max_event_gap_seconds,
            transition_penalty=transition_penalty_fn,
        )

    @staticmethod
    def _as_frame_candidate(hit: Any) -> FrameCandidate | None:
        metadata = hit.metadata
        video_id = metadata.get("video_id")
        frame = metadata.get("frame_idx", metadata.get("frame_id"))
        timestamp = metadata.get("timestamp", metadata.get("pts_time"))
        fps = metadata.get("fps")
        if video_id is None:
            return None
        if timestamp is None and frame is not None and fps:
            timestamp = float(frame) / float(fps)
        if frame is None and timestamp is not None and fps:
            frame = round(float(timestamp) * float(fps))
        if frame is None or timestamp is None:
            return None
        return FrameCandidate(
            candidate_id=hit.candidate_id,
            video_id=str(video_id),
            frame_idx=int(frame),
            timestamp=float(timestamp),
            score=float(hit.score),
            metadata=metadata,
        )
