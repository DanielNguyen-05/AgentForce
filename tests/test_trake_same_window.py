from agentforce.retrieval.types import SearchHit
from agentforce.tasks.trake import TRAKEConfig, TRAKESolver
from agentforce.temporal.refinement import FrameCandidate


class SeparatingRefiner:
    def refine(self, query, candidates):
        coarse = candidates[0]
        offset = 1 if "first" in query else 2
        return [
            FrameCandidate(
                candidate_id=f"{coarse.candidate_id}:{offset}",
                video_id=coarse.video_id,
                frame_idx=100 + offset,
                timestamp=10.0 + offset,
                score=coarse.score,
            )
        ]


class ReversingRefiner:
    def refine(self, query, candidates):
        coarse = candidates[0]
        timestamp = 12.0 if "first" in query else 11.0
        return [
            FrameCandidate(
                candidate_id=f"{coarse.candidate_id}:{query}",
                video_id=coarse.video_id,
                frame_idx=int(timestamp * 25),
                timestamp=timestamp,
                score=coarse.score,
            )
        ]


def test_dense_refinement_can_separate_events_from_same_coarse_window() -> None:
    solver = TRAKESolver(
        dense_refiner=SeparatingRefiner(),
        config=TRAKEConfig(top_k=5),
    )
    same_window = SearchHit(
        candidate_id="W1",
        score=0.9,
        metadata={"video_id": "L01_V001", "frame_idx": 100, "pts_time": 10.0},
    )
    answers = solver.solve_from_results(
        "1. first event\n2. second event",
        [{"visual": [same_window]}, {"visual": [same_window]}],
    )
    assert answers
    assert answers[0].frame_ids == (101, 102)


def test_dense_refinement_rejects_a_reversed_exact_frame_path() -> None:
    solver = TRAKESolver(
        dense_refiner=ReversingRefiner(),
        config=TRAKEConfig(top_k=5, strict_order=True),
    )
    same_window = SearchHit(
        candidate_id="W1",
        score=0.9,
        metadata={"video_id": "L01_V001", "frame_idx": 100, "pts_time": 10.0},
    )

    answers = solver.solve_from_results(
        "1. first event\n2. second event",
        [{"visual": [same_window]}, {"visual": [same_window]}],
    )

    assert answers == []
