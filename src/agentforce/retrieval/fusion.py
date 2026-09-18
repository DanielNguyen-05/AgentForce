"""Rank- and score-level fusion for independent retrieval modalities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal

from .types import FusedHit, SearchHit

Normalization = Literal["none", "minmax", "rank"]


def _deduplicate(hits: Sequence[SearchHit]) -> list[SearchHit]:
    best: dict[str, tuple[int, SearchHit]] = {}
    for position, hit in enumerate(hits, start=1):
        previous = best.get(hit.candidate_id)
        if previous is None or hit.score > previous[1].score:
            best[hit.candidate_id] = (position, hit)
    return [
        item[1]
        for item in sorted(
            best.values(), key=lambda pair: (-pair[1].score, pair[0], pair[1].candidate_id)
        )
    ]


def _metadata_for(existing: dict[str, object], hit: SearchHit) -> None:
    for key, value in hit.metadata.items():
        existing.setdefault(key, value)


def reciprocal_rank_fusion(
    results: Mapping[str, Sequence[SearchHit]],
    *,
    weights: Mapping[str, float] | None = None,
    rank_constant: float = 60.0,
    limit: int | None = None,
) -> list[FusedHit]:
    """Fuse ranked lists using weighted Reciprocal Rank Fusion (RRF).

    The keys in ``results`` identify independent sources, for example ``visual``
    or ``asr:translated``. Input ``SearchHit.rank`` is deliberately ignored when
    it is zero; the order supplied by each searcher is the source of truth.
    """

    if rank_constant < 0:
        raise ValueError("rank_constant must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    source_weights = weights or {}
    fused_scores: defaultdict[str, float] = defaultdict(float)
    modality_scores: defaultdict[str, dict[str, float]] = defaultdict(dict)
    modality_ranks: defaultdict[str, dict[str, int]] = defaultdict(dict)
    metadata: defaultdict[str, dict[str, object]] = defaultdict(dict)

    for source, source_hits in results.items():
        weight = float(source_weights.get(source, 1.0))
        if weight < 0:
            raise ValueError(f"weight for {source!r} must be non-negative")
        for rank, hit in enumerate(_deduplicate(source_hits), start=1):
            contribution = weight / (rank_constant + rank)
            fused_scores[hit.candidate_id] += contribution
            modality_scores[hit.candidate_id][source] = hit.score
            modality_ranks[hit.candidate_id][source] = rank
            _metadata_for(metadata[hit.candidate_id], hit)

    fused = [
        FusedHit(
            candidate_id=candidate_id,
            score=score,
            modality_scores=modality_scores[candidate_id],
            modality_ranks=modality_ranks[candidate_id],
            metadata=metadata[candidate_id],
        )
        for candidate_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda hit: (-hit.score, hit.candidate_id))
    return fused if limit is None else fused[:limit]


def _normalized_scores(
    hits: Sequence[SearchHit], normalization: Normalization
) -> dict[str, float]:
    deduplicated = _deduplicate(hits)
    if not deduplicated:
        return {}
    if normalization == "rank":
        return {
            hit.candidate_id: 1.0 / rank
            for rank, hit in enumerate(deduplicated, start=1)
        }
    if normalization == "none":
        return {hit.candidate_id: hit.score for hit in deduplicated}
    if normalization != "minmax":
        raise ValueError(f"unsupported normalization: {normalization}")
    values = [hit.score for hit in deduplicated]
    low, high = min(values), max(values)
    if high == low:
        return {hit.candidate_id: 1.0 for hit in deduplicated}
    return {
        hit.candidate_id: (hit.score - low) / (high - low) for hit in deduplicated
    }


def weighted_score_fusion(
    results: Mapping[str, Sequence[SearchHit]],
    *,
    weights: Mapping[str, float] | None = None,
    normalization: Normalization = "minmax",
    limit: int | None = None,
) -> list[FusedHit]:
    """Fuse source scores after optional per-source normalization."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    source_weights = weights or {}
    fused_scores: defaultdict[str, float] = defaultdict(float)
    modality_scores: defaultdict[str, dict[str, float]] = defaultdict(dict)
    modality_ranks: defaultdict[str, dict[str, int]] = defaultdict(dict)
    metadata: defaultdict[str, dict[str, object]] = defaultdict(dict)

    for source, source_hits in results.items():
        weight = float(source_weights.get(source, 1.0))
        if weight < 0:
            raise ValueError(f"weight for {source!r} must be non-negative")
        deduplicated = _deduplicate(source_hits)
        normalized = _normalized_scores(deduplicated, normalization)
        for rank, hit in enumerate(deduplicated, start=1):
            value = normalized[hit.candidate_id]
            fused_scores[hit.candidate_id] += weight * value
            modality_scores[hit.candidate_id][source] = hit.score
            modality_ranks[hit.candidate_id][source] = rank
            _metadata_for(metadata[hit.candidate_id], hit)

    fused = [
        FusedHit(
            candidate_id=candidate_id,
            score=score,
            modality_scores=modality_scores[candidate_id],
            modality_ranks=modality_ranks[candidate_id],
            metadata=metadata[candidate_id],
        )
        for candidate_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda hit: (-hit.score, hit.candidate_id))
    return fused if limit is None else fused[:limit]

