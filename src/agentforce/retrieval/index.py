"""Vector-index adapters with a deterministic NumPy exact-search baseline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Mapping, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .types import SearchHit

Metric = Literal["cosine", "inner_product", "l2"]
Backend = Literal["auto", "numpy", "faiss"]


class VectorIndex(Protocol):
    """Minimal interface shared by exact and approximate index adapters."""

    @property
    def dimension(self) -> int: ...

    def search(self, query: ArrayLike, k: int) -> list[SearchHit]: ...


def _as_matrix(vectors: ArrayLike) -> NDArray[np.float32]:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"vectors must be a 2-D matrix, got shape {matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("vectors must contain at least one row and one column")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors contain NaN or infinity")
    return np.ascontiguousarray(matrix)


def _as_query(query: ArrayLike, dimension: int) -> NDArray[np.float32]:
    vector = np.asarray(query, dtype=np.float32)
    if vector.ndim == 2 and vector.shape[0] == 1:
        vector = vector[0]
    if vector.ndim != 1 or vector.shape[0] != dimension:
        raise ValueError(
            f"query must have shape ({dimension},), got {tuple(vector.shape)}"
        )
    if not np.isfinite(vector).all():
        raise ValueError("query contains NaN or infinity")
    return np.ascontiguousarray(vector)


def _normalize_rows(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _normalize_vector(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else np.zeros_like(vector)


class NumpyExactIndex:
    """Exact vector search useful as the reference implementation and fallback.

    ``cosine`` normalizes both stored vectors and every query. ``inner_product``
    preserves the original scale. For ``l2`` the public score is negative squared
    distance, so every backend consistently treats larger scores as better.
    """

    def __init__(
        self,
        vectors: ArrayLike,
        ids: Sequence[str],
        *,
        metric: Metric = "cosine",
        metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        matrix = _as_matrix(vectors)
        if metric not in {"cosine", "inner_product", "l2"}:
            raise ValueError(f"unsupported metric: {metric}")
        if len(ids) != matrix.shape[0]:
            raise ValueError("ids length must equal number of vector rows")
        if len(set(ids)) != len(ids):
            raise ValueError("ids must be unique")
        if metadata is not None and len(metadata) != len(ids):
            raise ValueError("metadata length must equal ids length")

        self._metric = metric
        self._vectors = _normalize_rows(matrix) if metric == "cosine" else matrix
        self._ids = tuple(str(item) for item in ids)
        self._metadata = tuple(dict(item) for item in metadata) if metadata else None

    @property
    def dimension(self) -> int:
        return int(self._vectors.shape[1])

    @property
    def size(self) -> int:
        return int(self._vectors.shape[0])

    def search(self, query: ArrayLike, k: int) -> list[SearchHit]:
        if k <= 0:
            return []
        vector = _as_query(query, self.dimension)
        if self._metric == "cosine":
            vector = _normalize_vector(vector)
        if self._metric in {"cosine", "inner_product"}:
            scores = self._vectors @ vector
        else:
            delta = self._vectors - vector
            scores = -np.einsum("ij,ij->i", delta, delta)

        result_count = min(k, self.size)
        if result_count == self.size:
            indices = np.argsort(-scores, kind="stable")
        else:
            partition = np.argpartition(-scores, result_count - 1)[:result_count]
            indices = partition[np.argsort(-scores[partition], kind="stable")]

        return [
            SearchHit(
                candidate_id=self._ids[int(index)],
                score=float(scores[int(index)]),
                rank=rank,
                metadata=(self._metadata[int(index)] if self._metadata else {}),
            )
            for rank, index in enumerate(indices, start=1)
        ]


class FaissVectorIndex:
    """Optional FAISS flat index with the same result contract as NumPy."""

    def __init__(
        self,
        vectors: ArrayLike,
        ids: Sequence[str],
        *,
        metric: Metric = "cosine",
        metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        try:
            import faiss  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "FAISS is not installed; install faiss-cpu or use backend='numpy'"
            ) from exc

        matrix = _as_matrix(vectors)
        if len(ids) != matrix.shape[0]:
            raise ValueError("ids length must equal number of vector rows")
        if len(set(ids)) != len(ids):
            raise ValueError("ids must be unique")
        if metadata is not None and len(metadata) != len(ids):
            raise ValueError("metadata length must equal ids length")
        if metric not in {"cosine", "inner_product", "l2"}:
            raise ValueError(f"unsupported metric: {metric}")

        self._metric = metric
        if metric == "cosine":
            matrix = _normalize_rows(matrix)
        self._index = (
            faiss.IndexFlatL2(matrix.shape[1])
            if metric == "l2"
            else faiss.IndexFlatIP(matrix.shape[1])
        )
        self._index.add(matrix)
        self._ids = tuple(str(item) for item in ids)
        self._metadata = tuple(dict(item) for item in metadata) if metadata else None

    @property
    def dimension(self) -> int:
        return int(self._index.d)

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def search(self, query: ArrayLike, k: int) -> list[SearchHit]:
        if k <= 0:
            return []
        vector = _as_query(query, self.dimension)
        if self._metric == "cosine":
            vector = _normalize_vector(vector)
        distances, indices = self._index.search(vector[None, :], min(k, self.size))
        hits: list[SearchHit] = []
        for rank, (distance, index) in enumerate(
            zip(distances[0], indices[0], strict=True), start=1
        ):
            if index < 0:
                continue
            score = -float(distance) if self._metric == "l2" else float(distance)
            hits.append(
                SearchHit(
                    candidate_id=self._ids[int(index)],
                    score=score,
                    rank=rank,
                    metadata=(self._metadata[int(index)] if self._metadata else {}),
                )
            )
        return hits


def create_vector_index(
    vectors: ArrayLike,
    ids: Sequence[str],
    *,
    metric: Metric = "cosine",
    backend: Backend = "auto",
    metadata: Sequence[Mapping[str, Any]] | None = None,
) -> VectorIndex:
    """Build an index, falling back to NumPy when optional FAISS is unavailable."""

    if backend == "numpy":
        return NumpyExactIndex(vectors, ids, metric=metric, metadata=metadata)
    if backend == "faiss":
        return FaissVectorIndex(vectors, ids, metric=metric, metadata=metadata)
    if backend != "auto":
        raise ValueError(f"unsupported backend: {backend}")
    try:
        return FaissVectorIndex(vectors, ids, metric=metric, metadata=metadata)
    except ImportError:
        return NumpyExactIndex(vectors, ids, metric=metric, metadata=metadata)

