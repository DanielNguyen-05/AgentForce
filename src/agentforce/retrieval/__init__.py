"""Composable retrieval primitives for AgentForce."""

from .fusion import reciprocal_rank_fusion, weighted_score_fusion
from .index import FaissVectorIndex, NumpyExactIndex, VectorIndex, create_vector_index
from .query import (
    HeuristicQueryExpander,
    HeuristicQueryParser,
    IdentityQueryExpander,
    QueryExpander,
    QueryParser,
)
from .ranking import aggregate_by_video, diversify_candidates, to_retrieval_candidate
from .types import (
    FusedHit,
    ParsedQuery,
    QueryEvent,
    QueryVariant,
    RetrievalCandidate,
    SearchHit,
    TaskType,
    VideoAggregate,
)

__all__ = [
    "FaissVectorIndex",
    "FusedHit",
    "HeuristicQueryExpander",
    "HeuristicQueryParser",
    "IdentityQueryExpander",
    "NumpyExactIndex",
    "ParsedQuery",
    "QueryEvent",
    "QueryExpander",
    "QueryParser",
    "QueryVariant",
    "RetrievalCandidate",
    "SearchHit",
    "TaskType",
    "VectorIndex",
    "VideoAggregate",
    "aggregate_by_video",
    "create_vector_index",
    "diversify_candidates",
    "reciprocal_rank_fusion",
    "to_retrieval_candidate",
    "weighted_score_fusion",
]
