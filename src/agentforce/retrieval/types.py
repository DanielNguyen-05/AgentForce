"""Small, dependency-free data contracts used by retrieval and task solvers.

The canonical dataset records live outside this package.  Retrieval intentionally
accepts plain metadata mappings so an index can be backed by Parquet, SQLite, or
an in-memory fixture without changing the public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


class TaskType(StrEnum):
    """Supported AIC query families."""

    KIS = "kis"
    QA = "qa"
    TRAKE = "trake"


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a shallow immutable copy suitable for a frozen dataclass."""

    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One result returned by a modality-specific searcher."""

    candidate_id: str
    score: float
    rank: int = 0
    modality: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class FusedHit:
    """A candidate after combining ranks or scores from multiple sources."""

    candidate_id: str
    score: float
    modality_scores: Mapping[str, float] = field(default_factory=dict)
    modality_ranks: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(
            self, "modality_scores", MappingProxyType(dict(self.modality_scores))
        )
        object.__setattr__(
            self, "modality_ranks", MappingProxyType(dict(self.modality_ranks))
        )
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class QueryVariant:
    """A weighted textual representation of a query."""

    text: str
    weight: float = 1.0
    kind: str = "original"
    modality: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("query variant text must not be empty")
        if self.weight <= 0:
            raise ValueError("query variant weight must be positive")


@dataclass(frozen=True, slots=True)
class QueryEvent:
    """One ordered event in a TRAKE query."""

    order: int
    description: str

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("event order starts at one")
        if not self.description.strip():
            raise ValueError("event description must not be empty")


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    """Structured interpretation produced before retrieval."""

    raw_text: str
    task_type: TaskType
    retrieval_text: str
    question: str | None = None
    language: str = "unknown"
    expected_answer_type: str | None = None
    events: tuple[QueryEvent, ...] = ()
    modality_hints: Mapping[str, float] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality_hints", _frozen_mapping(self.modality_hints))
        object.__setattr__(self, "attributes", _frozen_mapping(self.attributes))


@dataclass(frozen=True, slots=True)
class VideoAggregate:
    """Ranked video assembled from its strongest local candidates."""

    video_id: str
    score: float
    candidates: tuple[FusedHit, ...]
    modality_count: int


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """Canonical frame-level candidate consumed by task solvers."""

    candidate_id: str
    video_id: str
    score: float
    frame_idx: int | None = None
    timestamp: float | None = None
    window_id: str | None = None
    modality_scores: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("video_id must not be empty")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(
            self, "modality_scores", MappingProxyType(dict(self.modality_scores))
        )
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

