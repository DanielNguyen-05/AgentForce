"""Memory-mapped exact cosine search used as the transparent baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
import heapq
import json

import numpy as np

from agentforce.embeddings.encoders import l2_normalize


@dataclass(frozen=True, slots=True)
class VectorHit:
    vector_id: str
    score: float
    row: int
    metadata: Mapping[str, Any]


class NumpyIndex:
    """Exact inner-product index over normalized embeddings.

    Search is chunked so a large memory-mapped index remains debuggable on a
    laptop. Supplied AIC CLIP features are already L2 normalized.
    """

    def __init__(
        self,
        vectors_path: str | Path,
        metadata_path: str | Path,
        *,
        chunk_size: int = 32_768,
    ) -> None:
        self.vectors_path = Path(vectors_path)
        self.metadata_path = Path(metadata_path)
        self.vectors = np.load(self.vectors_path, mmap_mode="r")
        if self.vectors.ndim != 2:
            raise ValueError(f"Expected 2D vector matrix, got {self.vectors.shape}")
        self.metadata = list(_read_jsonl(self.metadata_path))
        if len(self.metadata) != self.vectors.shape[0]:
            raise ValueError(
                f"Metadata rows ({len(self.metadata)}) do not match vectors ({self.vectors.shape[0]})"
            )
        self.chunk_size = int(chunk_size)
        self.dimension = int(self.vectors.shape[1])

    def search(self, query: np.ndarray, top_k: int = 10) -> list[VectorHit]:
        if top_k <= 0:
            return []
        vector = l2_normalize(np.asarray(query, dtype=np.float32)).reshape(-1)
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension {vector.shape[0]} does not match index {self.dimension}"
            )

        heap: list[tuple[float, int]] = []
        for start in range(0, len(self.metadata), self.chunk_size):
            stop = min(start + self.chunk_size, len(self.metadata))
            block = np.asarray(self.vectors[start:stop], dtype=np.float32)
            scores = block @ vector
            local_k = min(top_k, scores.size)
            if local_k == scores.size:
                indices = np.arange(scores.size)
            else:
                indices = np.argpartition(scores, -local_k)[-local_k:]
            for local_row in indices:
                item = (float(scores[local_row]), start + int(local_row))
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

        hits = []
        for score, row in sorted(heap, reverse=True):
            metadata = self.metadata[row]
            vector_id = str(metadata.get("vector_id", metadata.get("keyframe_uid", row)))
            hits.append(VectorHit(vector_id=vector_id, score=score, row=row, metadata=metadata))
        return hits


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            yield value


def write_metadata(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(output)
    return count

