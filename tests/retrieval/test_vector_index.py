import numpy as np
import pytest

from agentforce.retrieval.index import NumpyExactIndex, create_vector_index


def test_cosine_exact_search_returns_ranked_metadata() -> None:
    index = NumpyExactIndex(
        [[2.0, 0.0], [1.0, 1.0], [0.0, 3.0]],
        ["x", "xy", "y"],
        metadata=[{"row": 0}, {"row": 1}, {"row": 2}],
    )

    hits = index.search([1.0, 0.0], k=2)

    assert [hit.candidate_id for hit in hits] == ["x", "xy"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].metadata["row"] == 0
    assert hits[0].score == pytest.approx(1.0)


def test_l2_exposes_negative_distance_so_larger_is_better() -> None:
    index = NumpyExactIndex([[0.0], [2.0], [5.0]], ["a", "b", "c"], metric="l2")

    hits = index.search([1.5], k=3)

    assert [hit.candidate_id for hit in hits] == ["b", "a", "c"]
    assert hits[0].score == pytest.approx(-0.25)


def test_numpy_index_validates_shape_and_ids() -> None:
    with pytest.raises(ValueError, match="ids length"):
        NumpyExactIndex(np.ones((2, 3)), ["only-one"])
    with pytest.raises(ValueError, match="query must have shape"):
        NumpyExactIndex(np.ones((2, 3)), ["a", "b"]).search([1.0, 2.0], 1)


def test_index_factory_always_has_numpy_backend() -> None:
    index = create_vector_index([[1.0, 0.0]], ["a"], backend="numpy")
    assert index.search([1.0, 0.0], 1)[0].candidate_id == "a"

