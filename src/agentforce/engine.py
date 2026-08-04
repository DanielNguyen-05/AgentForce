"""High-level multimodal retrieval orchestration.

The engine contains no model-specific logic. Encoders and indexes are injected,
which makes every retrieval stage independently testable and allows production
FAISS indexes to replace transparent NumPy indexes without changing task code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

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


class IndexTextSearcher:
    """Adapter used by task solvers that own fusion and temporal logic."""

    def __init__(self, field: SearchField) -> None:
        self.field = field

    def search(self, text: str, *, k: int) -> Sequence[SearchHit]:
        query = self.field.encoder.encode([text])[0]
        return tuple(
            SearchHit(
                candidate_id=hit.vector_id,
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
                previous = best.get(hit.vector_id)
                if previous is None or score > previous.score:
                    best[hit.vector_id] = SearchHit(
                        candidate_id=hit.vector_id,
                        score=score,
                        rank=rank,
                        modality=field.name,
                        metadata=hit.metadata,
                    )
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
