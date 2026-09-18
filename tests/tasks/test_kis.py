from agentforce.retrieval.types import SearchHit
from agentforce.tasks.kis import KISConfig, KISSolver


def test_kis_solver_fuses_and_emits_exact_frame_ids() -> None:
    results = {
        "visual": [
            SearchHit("a", 0.9, metadata={"video_id": "v1", "frame_idx": 25, "timestamp": 1.0}),
            SearchHit("b", 0.8, metadata={"video_id": "v2", "frame_idx": 50, "timestamp": 2.0}),
        ],
        "asr": [
            SearchHit("b", 4.0, metadata={"video_id": "v2", "frame_idx": 50, "timestamp": 2.0}),
        ],
    }
    solver = KISSolver(config=KISConfig(top_k=2, fusion_pool_size=10))

    answers = solver.solve_from_results("một người đang phát biểu", results)

    assert answers[0].video_id == "v2"
    assert answers[0].frame_idx == 50
    assert set(answers[0].evidence["modality_scores"]) == {"visual", "asr"}


def test_kis_solver_skips_candidates_without_frame_mapping() -> None:
    solver = KISSolver(config=KISConfig(top_k=1, fusion_pool_size=2))
    answers = solver.solve_from_results(
        "cảnh ngoài trời",
        {"ocr": [SearchHit("window", 1.0, metadata={"video_id": "v1"})]},
    )
    assert answers == []
