"""End-to-end orchestration for Textual Known-Item Search (KIS)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentforce.retrieval.query import (
    HeuristicQueryParser,
    IdentityQueryExpander,
    QueryExpander,
    QueryParser,
)
from agentforce.retrieval.ranking import diversify_candidates, to_retrieval_candidate
from agentforce.retrieval.types import RetrievalCandidate, SearchHit, TaskType
from agentforce.temporal.refinement import DenseRefiner, FrameCandidate

from .common import CandidateReranker, TextSearcher, collect_text_results, fuse


@dataclass(frozen=True, slots=True)
class KISConfig:
    per_source_k: int = 500
    fusion_pool_size: int = 1_000
    top_k: int = 100
    fusion_method: str = "rrf"
    rank_constant: float = 60.0
    score_normalization: str = "minmax"
    modality_weights: Mapping[str, float] = field(default_factory=dict)
    same_video_penalty: float = 0.05
    temporal_penalty: float = 0.3
    temporal_radius_seconds: float = 2.0
    max_per_video: int | None = None
    dense_refine_top_n: int = 0

    def __post_init__(self) -> None:
        if min(self.per_source_k, self.fusion_pool_size, self.top_k) <= 0:
            raise ValueError("retrieval sizes must be positive")
        if self.dense_refine_top_n < 0:
            raise ValueError("dense_refine_top_n must be non-negative")


@dataclass(frozen=True, slots=True)
class KISAnswer:
    video_id: str
    frame_idx: int
    score: float
    candidate_id: str
    timestamp: float | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)


class KISSolver:
    """Retrieve, fuse, optionally rerank/refine, and return exact frame IDs."""

    def __init__(
        self,
        searchers: Mapping[str, TextSearcher] | None = None,
        *,
        parser: QueryParser | None = None,
        expander: QueryExpander | None = None,
        reranker: CandidateReranker | None = None,
        dense_refiner: DenseRefiner | None = None,
        config: KISConfig | None = None,
    ) -> None:
        self._searchers = dict(searchers or {})
        self._parser = parser or HeuristicQueryParser()
        self._expander = expander or IdentityQueryExpander()
        self._reranker = reranker
        self._dense_refiner = dense_refiner
        self._config = config or KISConfig()

    def solve(self, query: str) -> list[KISAnswer]:
        if not self._searchers:
            raise RuntimeError("KISSolver.solve requires at least one text searcher")
        parsed = self._parser.parse(query, task_type=TaskType.KIS)
        results, weights = collect_text_results(
            parsed,
            searchers=self._searchers,
            expander=self._expander,
            per_source_k=self._config.per_source_k,
            modality_weights=self._config.modality_weights,
        )
        return self._solve_fused(parsed.retrieval_text, results, weights)

    def solve_from_results(
        self,
        query: str,
        results: Mapping[str, Sequence[SearchHit]],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> list[KISAnswer]:
        """Run deterministic downstream logic on cached/precomputed search output."""

        parsed = self._parser.parse(query, task_type=TaskType.KIS)
        resolved_weights = {
            source: float(
                (weights or {}).get(
                    source,
                    self._config.modality_weights.get(source.split(":", 1)[0], 1.0)
                    * parsed.modality_hints.get(source.split(":", 1)[0], 1.0),
                )
            )
            for source in results
        }
        return self._solve_fused(parsed.retrieval_text, results, resolved_weights)

    def _solve_fused(
        self,
        query: str,
        results: Mapping[str, Sequence[SearchHit]],
        weights: Mapping[str, float],
    ) -> list[KISAnswer]:
        fused = fuse(
            results,
            method=self._config.fusion_method,
            weights=weights,
            limit=self._config.fusion_pool_size,
            rank_constant=self._config.rank_constant,
            score_normalization=self._config.score_normalization,
        )
        diversified = diversify_candidates(
            fused,
            limit=self._config.fusion_pool_size,
            same_video_penalty=self._config.same_video_penalty,
            temporal_penalty=self._config.temporal_penalty,
            temporal_radius_seconds=self._config.temporal_radius_seconds,
            max_per_video=self._config.max_per_video,
        )
        candidates: list[RetrievalCandidate] = []
        for hit in diversified:
            try:
                candidate = to_retrieval_candidate(hit)
            except (TypeError, ValueError):
                continue
            if candidate.frame_idx is not None:
                candidates.append(candidate)
        if self._reranker:
            candidates = list(self._reranker.rerank(query, candidates))
        candidates = self._apply_dense_refinement(query, candidates)

        answers: list[KISAnswer] = []
        seen: set[tuple[str, int]] = set()
        for candidate in candidates:
            assert candidate.frame_idx is not None
            key = (candidate.video_id, candidate.frame_idx)
            if key in seen:
                continue
            seen.add(key)
            answers.append(
                KISAnswer(
                    video_id=candidate.video_id,
                    frame_idx=candidate.frame_idx,
                    score=candidate.score,
                    candidate_id=candidate.candidate_id,
                    timestamp=candidate.timestamp,
                    evidence={
                        **candidate.metadata,
                        "modality_scores": dict(candidate.modality_scores),
                    },
                )
            )
            if len(answers) >= self._config.top_k:
                break
        return answers

    def _apply_dense_refinement(
        self, query: str, candidates: Sequence[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        count = min(self._config.dense_refine_top_n, len(candidates))
        if not self._dense_refiner or count == 0:
            return list(candidates)
        coarse_frames: list[FrameCandidate] = []
        passthrough: list[RetrievalCandidate] = []
        for index, candidate in enumerate(candidates):
            if index >= count or candidate.timestamp is None or candidate.frame_idx is None:
                passthrough.append(candidate)
                continue
            coarse_frames.append(
                FrameCandidate(
                    candidate_id=candidate.candidate_id,
                    video_id=candidate.video_id,
                    frame_idx=candidate.frame_idx,
                    timestamp=candidate.timestamp,
                    score=candidate.score,
                    metadata=candidate.metadata,
                )
            )
        refined = self._dense_refiner.refine(query, coarse_frames)
        converted = [
            RetrievalCandidate(
                candidate_id=item.candidate_id,
                video_id=item.video_id,
                frame_idx=item.frame_idx,
                timestamp=item.timestamp,
                score=item.score,
                metadata=item.metadata,
            )
            for item in refined
        ]
        return sorted(converted + passthrough, key=lambda item: (-item.score, item.candidate_id))

