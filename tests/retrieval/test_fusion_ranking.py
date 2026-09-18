from agentforce.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from agentforce.retrieval.ranking import aggregate_by_video, diversify_candidates
from agentforce.retrieval.types import FusedHit, SearchHit


def _hit(candidate_id: str, score: float, video: str, timestamp: float = 0.0) -> SearchHit:
    return SearchHit(
        candidate_id,
        score,
        metadata={"video_id": video, "timestamp": timestamp, "frame_idx": int(timestamp * 25)},
    )


def test_rrf_rewards_cross_modal_agreement() -> None:
    fused = reciprocal_rank_fusion(
        {
            "visual": [_hit("a", 0.9, "v1"), _hit("b", 0.8, "v2")],
            "asr": [_hit("b", 12.0, "v2"), _hit("c", 11.0, "v3")],
        },
        rank_constant=0,
    )
    assert fused[0].candidate_id == "b"
    assert set(fused[0].modality_scores) == {"visual", "asr"}


def test_weighted_fusion_normalizes_each_source() -> None:
    fused = weighted_score_fusion(
        {
            "visual": [_hit("a", 0.9, "v1"), _hit("b", 0.1, "v2")],
            "ocr": [_hit("b", 100.0, "v2"), _hit("a", 0.0, "v1")],
        },
        weights={"ocr": 2.0, "visual": 1.0},
    )
    assert fused[0].candidate_id == "b"


def test_video_aggregation_and_temporal_diversification() -> None:
    candidates = [
        FusedHit("a1", 1.0, {"visual": 1.0}, {}, {"video_id": "a", "timestamp": 1.0}),
        FusedHit("a2", 0.99, {"visual": 0.9}, {}, {"video_id": "a", "timestamp": 1.1}),
        FusedHit("b1", 0.95, {"asr": 0.8}, {}, {"video_id": "b", "timestamp": 5.0}),
    ]

    aggregates = aggregate_by_video(candidates)
    diversified = diversify_candidates(
        candidates,
        limit=3,
        same_video_penalty=0.1,
        temporal_penalty=0.8,
        temporal_radius_seconds=2.0,
    )

    assert aggregates[0].video_id == "a"
    assert [item.candidate_id for item in diversified[:2]] == ["a1", "b1"]

