import numpy as np

from agentforce.temporal.alignment import align_ordered_candidates, ordered_viterbi
from agentforce.temporal.refinement import FrameCandidate


def _candidate(event: int, time: float, score: float) -> FrameCandidate:
    return FrameCandidate(
        candidate_id=f"e{event}:{time}",
        video_id="video",
        frame_idx=int(time * 25),
        timestamp=time,
        score=score,
    )


def test_ordered_viterbi_rejects_high_scoring_reversed_pair() -> None:
    scores = np.asarray([[10.0, 1.0, 9.0, 0.0], [0.0, 10.0, 0.0, 8.0]])

    path = ordered_viterbi(scores, [0.0, 1.0, 2.0, 3.0])

    assert path is not None
    assert path.indices == (0, 1)
    assert path.total_score == 20.0


def test_ordered_viterbi_honors_max_gap() -> None:
    path = ordered_viterbi([[1.0, -np.inf], [-np.inf, 1.0]], [0.0, 10.0], max_gap=5.0)
    assert path is None


def test_ordered_viterbi_reports_the_penalized_score() -> None:
    path = ordered_viterbi(
        [[2.0, -np.inf], [-np.inf, 3.0]],
        [0.0, 4.0],
        transition_penalty=lambda gap: gap * 0.25,
    )

    assert path is not None
    assert path.total_score == 4.0


def test_alignment_supports_different_candidate_timelines() -> None:
    alignment = align_ordered_candidates(
        [
            [_candidate(1, 20.0, 10.0), _candidate(1, 5.0, 8.0)],
            [_candidate(2, 10.0, 9.0), _candidate(2, 30.0, 7.0)],
        ]
    )

    assert alignment is not None
    assert list(alignment.event_timestamps if hasattr(alignment, "event_timestamps") else [c.timestamp for c in alignment.candidates]) == [5.0, 10.0]
    assert alignment.frame_ids == (125, 250)
