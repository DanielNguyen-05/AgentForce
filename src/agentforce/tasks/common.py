"""Shared orchestration helpers for task-specific solvers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from agentforce.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from agentforce.retrieval.query import QueryExpander
from agentforce.retrieval.types import FusedHit, ParsedQuery, RetrievalCandidate, SearchHit


class TextSearcher(Protocol):
    """A modality adapter that owns text encoding and its underlying index."""

    def search(self, text: str, *, k: int) -> Sequence[SearchHit]: ...


class CandidateReranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[RetrievalCandidate]
    ) -> Sequence[RetrievalCandidate]: ...


def canonicalize_hit(hit: SearchHit, *, modality: str) -> SearchHit:
    """Map modality-specific IDs onto an optional cross-modal ``fusion_id``."""

    fusion_id = str(hit.metadata.get("fusion_id", hit.candidate_id))
    metadata = dict(hit.metadata)
    if fusion_id != hit.candidate_id:
        metadata.setdefault(f"{modality}_candidate_id", hit.candidate_id)
    return SearchHit(
        candidate_id=fusion_id,
        score=hit.score,
        rank=hit.rank,
        modality=modality,
        metadata=metadata,
    )


def collect_text_results(
    query: ParsedQuery,
    *,
    searchers: Mapping[str, TextSearcher],
    expander: QueryExpander,
    per_source_k: int,
    modality_weights: Mapping[str, float],
) -> tuple[dict[str, Sequence[SearchHit]], dict[str, float]]:
    """Search every allowed modality/variant and return source-specific weights."""

    results: dict[str, Sequence[SearchHit]] = {}
    source_weights: dict[str, float] = {}
    variants = expander.expand(query)
    for modality, searcher in searchers.items():
        hint = float(query.modality_hints.get(modality, 1.0))
        modality_weight = float(modality_weights.get(modality, 1.0))
        for variant_index, variant in enumerate(variants):
            if variant.modality is not None and variant.modality != modality:
                continue
            source = f"{modality}:{variant.kind}:{variant_index}"
            hits = searcher.search(variant.text, k=per_source_k)
            results[source] = tuple(
                canonicalize_hit(hit, modality=modality) for hit in hits
            )
            source_weights[source] = modality_weight * hint * variant.weight
    return results, source_weights


def fuse(
    results: Mapping[str, Sequence[SearchHit]],
    *,
    method: str,
    weights: Mapping[str, float],
    limit: int,
    rank_constant: float,
    score_normalization: str,
) -> list[FusedHit]:
    if method == "rrf":
        return reciprocal_rank_fusion(
            results,
            weights=weights,
            rank_constant=rank_constant,
            limit=limit,
        )
    if method == "weighted":
        return weighted_score_fusion(
            results,
            weights=weights,
            normalization=score_normalization,  # type: ignore[arg-type]
            limit=limit,
        )
    raise ValueError(f"unsupported fusion method: {method}")

