from collections.abc import Iterable, Sequence

from agentforce.temporal.refinement import DecodedFrame, DenseFrameRefiner, FrameCandidate


class FakeSource:
    def iter_frames(
        self,
        video_id: str,
        *,
        start_time: float,
        end_time: float,
        sample_fps: float,
    ) -> Iterable[DecodedFrame]:
        del start_time, end_time, sample_fps
        return [
            DecodedFrame(video_id, 99, 3.96, "before"),
            DecodedFrame(video_id, 100, 4.00, "exact"),
            DecodedFrame(video_id, 101, 4.04, "after"),
        ]


class FakeScorer:
    def score(self, query: str, frames: Sequence[DecodedFrame]) -> Sequence[float]:
        del query
        return [1.0 if frame.payload == "exact" else 0.1 for frame in frames]


def test_dense_refiner_returns_decoder_frame_ids() -> None:
    refiner = DenseFrameRefiner(FakeSource(), FakeScorer(), top_per_candidate=1)
    coarse = FrameCandidate("coarse", "video", 90, 3.6, 0.5)

    refined = refiner.refine("event", [coarse])

    assert len(refined) == 1
    assert refined[0].frame_idx == 100
    assert refined[0].source_candidate_id == "coarse"

