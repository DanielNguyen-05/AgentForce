"""Optional FAISS inner-product backend with the same hit contract as NumPy."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from agentforce.embeddings.encoders import l2_normalize
from agentforce.errors import OptionalDependencyError
from .numpy_index import VectorHit


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise OptionalDependencyError(
            "FAISS backend requires `pip install -e '.[faiss]'`; use NumpyIndex for debugging"
        ) from exc
    return faiss


def build_faiss_index(vectors_path: str | Path, output_path: str | Path) -> Path:
    faiss = _faiss()
    vectors = np.load(Path(vectors_path), mmap_mode="r")
    if vectors.ndim != 2:
        raise ValueError("FAISS builder expects a two-dimensional matrix")
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    for start in range(0, vectors.shape[0], 32_768):
        block = l2_normalize(np.asarray(vectors[start : start + 32_768], dtype=np.float32))
        index.add(block)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output))
    return output


class FaissIndex:
    def __init__(self, index_path: str | Path, metadata_path: str | Path) -> None:
        faiss = _faiss()
        self.index = faiss.read_index(str(index_path))
        with Path(metadata_path).open(encoding="utf-8") as handle:
            self.metadata = [json.loads(line) for line in handle if line.strip()]
        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS vector count does not match metadata row count")
        self.dimension = int(self.index.d)

    def search(self, query: np.ndarray, top_k: int = 10) -> list[VectorHit]:
        if top_k <= 0:
            return []
        vector = l2_normalize(query)
        if vector.shape[1] != self.dimension:
            raise ValueError("Query dimension does not match FAISS index")
        scores, rows = self.index.search(vector.astype(np.float32), min(top_k, len(self.metadata)))
        hits: list[VectorHit] = []
        for score, row in zip(scores[0], rows[0]):
            if row < 0:
                continue
            metadata: dict[str, Any] = self.metadata[int(row)]
            hits.append(
                VectorHit(
                    vector_id=str(metadata.get("vector_id", row)),
                    score=float(score),
                    row=int(row),
                    metadata=metadata,
                )
            )
        return hits

