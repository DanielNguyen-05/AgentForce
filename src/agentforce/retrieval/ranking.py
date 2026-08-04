"""Video aggregation, candidate conversion, and temporal diversification."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from .types import FusedHit, RetrievalCandidate, VideoAggregate


def to_retrieval_candidate(hit: FusedHit) -> RetrievalCandidate:
    """Convert a fused hit while validating submission-critical metadata."""

    metadata = hit.metadata
    video_id = metadata.get("video_id")
    if video_id is None or not str(video_id):
        raise ValueError(f"candidate {hit.candidate_id!r} has no video_id")
    frame_value = metadata.get("frame_idx", metadata.get("frame_id"))
    time_value = metadata.get("timestamp", metadata.get("pts_time"))
    return RetrievalCandidate(
        candidate_id=hit.candidate_id,
        video_id=str(video_id),
        score=hit.score,
        frame_idx=int(frame_value) if frame_value is not None else None,
        timestamp=float(time_value) if time_value is not None else None,
        window_id=(str(metadata["window_id"]) if metadata.get("window_id") else None),
        modality_scores=hit.modality_scores,
        metadata=metadata,
    )


def aggregate_by_video(
    candidates: Sequence[FusedHit],
    *,
    top_n: int = 3,
    max_score_weight: float = 0.65,
    mean_score_weight: float = 0.35,
    modality_agreement_bonus: float = 0.05,
    video_id_key: str = "video_id",
) -> list[VideoAggregate]:
    """Aggregate local evidence before selecting videos.

    Agreement is the fraction of all observed source names that support any of a
    video's top candidates. This keeps the bonus bounded regardless of how many
    retrieval sources are enabled.
    """

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if max_score_weight < 0 or mean_score_weight < 0 or modality_agreement_bonus < 0:
        raise ValueError("aggregation weights must be non-negative")

    grouped: defaultdict[str, list[FusedHit]] = defaultdict(list)
    all_modalities: set[str] = set()
    for candidate in candidates:
        video_id = candidate.metadata.get(video_id_key)
        if video_id is None:
            continue
        grouped[str(video_id)].append(candidate)
        all_modalities.update(candidate.modality_scores)

    aggregates: list[VideoAggregate] = []
    modality_denominator = max(1, len(all_modalities))
    for video_id, video_candidates in grouped.items():
        ranked = sorted(video_candidates, key=lambda hit: (-hit.score, hit.candidate_id))
        strongest = ranked[:top_n]
        maximum = strongest[0].score
        mean = sum(hit.score for hit in strongest) / len(strongest)
        modalities = {name for hit in strongest for name in hit.modality_scores}
        agreement = len(modalities) / modality_denominator
        score = (
            max_score_weight * maximum
            + mean_score_weight * mean
            + modality_agreement_bonus * agreement
        )
        aggregates.append(
            VideoAggregate(
                video_id=video_id,
                score=score,
                candidates=tuple(ranked),
                modality_count=len(modalities),
            )
        )
    aggregates.sort(key=lambda item: (-item.score, item.video_id))
    return aggregates


def _normalized_relevance(candidates: Sequence[FusedHit]) -> dict[str, float]:
    if not candidates:
        return {}
    values = [candidate.score for candidate in candidates]
    low, high = min(values), max(values)
    if low == high:
        return {candidate.candidate_id: 1.0 for candidate in candidates}
    return {
        candidate.candidate_id: (candidate.score - low) / (high - low)
        for candidate in candidates
    }


def _timestamp(metadata: dict[str, Any] | Any) -> float | None:
    value = metadata.get("timestamp", metadata.get("pts_time"))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def diversify_candidates(
    candidates: Sequence[FusedHit],
    *,
    limit: int = 100,
    same_video_penalty: float = 0.08,
    temporal_penalty: float = 0.35,
    temporal_radius_seconds: float = 2.0,
    max_per_video: int | None = None,
) -> list[FusedHit]:
    """Greedily diversify a ranked list while retaining its original scores.

    Selection uses normalized relevance minus penalties for repeated videos and
    near-duplicate timestamps. The returned objects keep their retrieval scores;
    only their order changes, making score logs honest and easy to debug.
    """

    if limit <= 0:
        return []
    if min(same_video_penalty, temporal_penalty, temporal_radius_seconds) < 0:
        raise ValueError("diversification penalties and radius must be non-negative")
    if max_per_video is not None and max_per_video <= 0:
        raise ValueError("max_per_video must be positive")

    remaining = list(candidates)
    relevance = _normalized_relevance(remaining)
    selected: list[FusedHit] = []
    video_counts: Counter[str] = Counter()

    while remaining and len(selected) < limit:
        best_index = -1
        best_key: tuple[float, float, str] | None = None
        for index, candidate in enumerate(remaining):
            video_id = str(candidate.metadata.get("video_id", ""))
            if max_per_video is not None and video_counts[video_id] >= max_per_video:
                continue
            utility = relevance[candidate.candidate_id]
            if video_counts[video_id]:
                utility -= same_video_penalty * video_counts[video_id]
            candidate_time = _timestamp(candidate.metadata)
            if candidate_time is not None and temporal_radius_seconds > 0:
                closeness = 0.0
                for previous in selected:
                    if str(previous.metadata.get("video_id", "")) != video_id:
                        continue
                    previous_time = _timestamp(previous.metadata)
                    if previous_time is None:
                        continue
                    distance = abs(candidate_time - previous_time)
                    closeness = max(
                        closeness,
                        max(0.0, 1.0 - distance / temporal_radius_seconds),
                    )
                utility -= temporal_penalty * closeness
            key = (utility, candidate.score, candidate.candidate_id)
            if best_key is None or key > best_key:
                best_key, best_index = key, index

        if best_index < 0:
            break
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        video_counts[str(chosen.metadata.get("video_id", ""))] += 1
    return selected
