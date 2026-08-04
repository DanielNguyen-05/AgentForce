import json

import numpy as np

from agentforce.indexing.numpy_index import NumpyIndex


def test_numpy_index_returns_cosine_neighbors(tmp_path) -> None:
    vectors = np.asarray([[1, 0], [0, 1], [0.8, 0.6]], dtype=np.float32)
    np.save(tmp_path / "vectors.npy", vectors)
    with (tmp_path / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(3):
            handle.write(json.dumps({"vector_id": f"v{index}"}) + "\n")
    index = NumpyIndex(tmp_path / "vectors.npy", tmp_path / "metadata.jsonl", chunk_size=2)
    hits = index.search(np.asarray([1, 0], dtype=np.float32), top_k=2)
    assert [hit.vector_id for hit in hits] == ["v0", "v2"]

