from agentforce.retrieval.types import SearchHit
from agentforce.tasks.trake import TRAKEConfig, TRAKESolver


def _hit(name: str, video: str, frame: int, timestamp: float, score: float) -> SearchHit:
    return SearchHit(
        name,
        score,
        metadata={
            "video_id": video,
            "frame_idx": frame,
            "timestamp": timestamp,
            "fps": 25.0,
        },
    )


def test_trake_solver_enforces_order_within_same_video() -> None:
    event_results = [
        {"visual": [_hit("late-e1", "v1", 500, 20.0, 1.0), _hit("early-e1", "v1", 125, 5.0, 0.8)]},
        {"visual": [_hit("early-e2", "v1", 250, 10.0, 1.0), _hit("late-e2", "v1", 750, 30.0, 0.8)]},
    ]
    solver = TRAKESolver(
        config=TRAKEConfig(top_k=2, top_videos=5, fusion_pool_size=10)
    )

    answers = solver.solve_from_results(
        "vận động viên giậm nhảy, sau đó tiếp đất", event_results
    )

    assert len(answers) == 1
    assert answers[0].video_id == "v1"
    assert answers[0].event_timestamps[0] < answers[0].event_timestamps[1]
    assert len(answers[0].frame_ids) == 2


def test_trake_requires_every_event_in_the_same_video() -> None:
    event_results = [
        {"visual": [_hit("e1", "v1", 25, 1.0, 1.0)]},
        {"visual": [_hit("e2", "v2", 50, 2.0, 1.0)]},
    ]
    answers = TRAKESolver().solve_from_results(
        "mở cửa rồi bước vào", event_results
    )
    assert answers == []

