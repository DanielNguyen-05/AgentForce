"""High-level multimodal retrieval orchestration.

The engine contains no model-specific logic. Encoders and indexes are injected,
which makes every retrieval stage independently testable and allows production
FAISS indexes to replace transparent NumPy indexes without changing task code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agentforce.embeddings.encoders import TextEncoder
from agentforce.indexing.numpy_index import NumpyIndex
from agentforce.retrieval.fusion import reciprocal_rank_fusion
from agentforce.retrieval.query import HeuristicQueryExpander, HeuristicQueryParser
from agentforce.retrieval.ranking import diversify_candidates
from agentforce.retrieval.types import FusedHit, ParsedQuery, QueryVariant, SearchHit, TaskType


@dataclass(slots=True)
class SearchField:
    name: str
    index: NumpyIndex
    encoder: TextEncoder
    top_k: int
    weight: float = 1.0


def canonical_fusion_id(vector_id: str, metadata: Mapping[str, Any]) -> str:
    """Resolve one cross-modal candidate ID from trusted index metadata.

    Window indexes use ``*_W...`` vector IDs while visual keyframes use
    ``*_K...`` IDs. Both point at the same moment through the representative
    keyframe UID, which therefore has priority for fusion.
    """

    for key in ("representative_keyframe_uid", "keyframe_uid"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    if not str(vector_id).strip():
        raise ValueError("vector_id must not be empty")
    return str(vector_id).strip()


def _canonical_search_hit(
    *,
    vector_id: str,
    score: float,
    rank: int,
    modality: str,
    metadata: Mapping[str, Any],
) -> SearchHit:
    """Build a canonical hit without losing source row provenance/context."""

    preserved = dict(metadata)
    preserved["vector_id"] = vector_id
    preserved.setdefault(f"{modality}_vector_id", vector_id)
    window_id = preserved.get("window_id")
    if window_id is not None and str(window_id).strip():
        preserved.setdefault(f"{modality}_window_id", str(window_id))
    context = preserved.get("text")
    if context is not None and str(context).strip():
        preserved.setdefault(f"{modality}_text", context)
    return SearchHit(
        candidate_id=canonical_fusion_id(vector_id, preserved),
        score=score,
        rank=rank,
        modality=modality,
        metadata=preserved,
    )


class IndexTextSearcher:
    """Adapter used by task solvers that own fusion and temporal logic."""

    def __init__(self, field: SearchField) -> None:
        self.field = field

    def search(self, text: str, *, k: int) -> Sequence[SearchHit]:
        query = self.field.encoder.encode([text])[0]
        return tuple(
            _canonical_search_hit(
                vector_id=hit.vector_id,
                score=hit.score,
                rank=rank,
                modality=self.field.name,
                metadata=hit.metadata,
            )
            for rank, hit in enumerate(self.field.index.search(query, k), 1)
        )


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: ParsedQuery
    variants: tuple[QueryVariant, ...]
    hits: tuple[FusedHit, ...]
    source_counts: Mapping[str, int]


class MultimodalSearchEngine:
    def __init__(
        self,
        fields: Sequence[SearchField],
        *,
        rank_constant: float = 60.0,
        duplicate_seconds: float = 2.0,
    ) -> None:
        if not fields:
            raise ValueError("At least one retrieval field is required")
        names = [field.name for field in fields]
        if len(names) != len(set(names)):
            raise ValueError("Retrieval field names must be unique")
        self.fields = tuple(fields)
        self.rank_constant = rank_constant
        self.duplicate_seconds = duplicate_seconds
        self.parser = HeuristicQueryParser()
        self.expander = HeuristicQueryExpander()

    @staticmethod
    def _search_field(field: SearchField, variants: Sequence[QueryVariant]) -> list[SearchHit]:
        # Keep the strongest score for a canonical candidate across translations
        # and paraphrases. Variant weights prevent loose paraphrases dominating.
        best: dict[str, SearchHit] = {}
        for variant in variants:
            query = field.encoder.encode([variant.text])[0]
            for rank, hit in enumerate(field.index.search(query, field.top_k), 1):
                score = hit.score * variant.weight
                candidate = _canonical_search_hit(
                    vector_id=hit.vector_id,
                    score=score,
                    rank=rank,
                    modality=field.name,
                    metadata=hit.metadata,
                )
                previous = best.get(candidate.candidate_id)
                if previous is None or score > previous.score:
                    best[candidate.candidate_id] = candidate
        return sorted(best.values(), key=lambda item: (-item.score, item.candidate_id))

    def search(
        self,
        text: str,
        *,
        task_type: TaskType | str | None = None,
        limit: int = 100,
    ) -> SearchResult:
        parsed = self.parser.parse(text, task_type=task_type)
        variants = tuple(self.expander.expand(parsed))
        source_results: dict[str, list[SearchHit]] = {}
        weights: dict[str, float] = {}
        for field in self.fields:
            source_results[field.name] = self._search_field(field, variants)
            hint = float(parsed.modality_hints.get(field.name, 1.0))
            weights[field.name] = field.weight * hint
        fused = reciprocal_rank_fusion(
            source_results,
            weights=weights,
            rank_constant=self.rank_constant,
        )
        diversified = diversify_candidates(
            fused,
            limit=limit,
            temporal_radius_seconds=self.duplicate_seconds,
        )
        return SearchResult(
            query=parsed,
            variants=variants,
            hits=tuple(diversified),
            source_counts={name: len(hits) for name, hits in source_results.items()},
        )
