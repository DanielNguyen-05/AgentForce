"""Bridge local retrieval results to cached Gemini multi-frame Q&A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from agentforce.data.schemas import DatasetManifest, KeyframeRecord
from agentforce.data.timeline import timeline_from_manifest
from agentforce.gemini import FrameCandidate, QAVerification, QAVerificationRequest, QAVerifier


@dataclass(frozen=True, slots=True)
class QAConfig:
    max_candidates: int = 12
    language: str = "vi"
    batch_size: int = 4

    def __post_init__(self) -> None:
        if not 1 <= self.max_candidates <= 64:
            raise ValueError("max_candidates must be between 1 and 64")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True, slots=True)
class RankedQAAnswer:
    result: QAVerification
    score: float


class QACandidateBuilder:
    """Resolve trusted local frame paths/IDs; never trust model-generated IDs."""

    def __init__(self, manifest: DatasetManifest) -> None:
        self.manifest = manifest
        self._timelines: dict[str, list[KeyframeRecord]] = {}

    def _timeline(self, video_id: str) -> list[KeyframeRecord]:
        if video_id not in self._timelines:
            self._timelines[video_id] = timeline_from_manifest(self.manifest, video_id)
        return self._timelines[video_id]

    def _resolve_record(self, video_id: str, metadata: dict[str, Any]) -> KeyframeRecord | None:
        timeline = self._timeline(video_id)
        uid = metadata.get("representative_keyframe_uid", metadata.get("keyframe_uid"))
        if uid:
            record = next((item for item in timeline if item.keyframe_uid == str(uid)), None)
            if record is not None:
                return record
        frame = metadata.get("frame_idx", metadata.get("frame_id"))
        if frame is not None:
            return min(timeline, key=lambda item: abs(item.frame_idx - int(frame)), default=None)
        timestamp = metadata.get("pts_time", metadata.get("timestamp"))
        if timestamp is not None:
            return min(timeline, key=lambda item: abs(item.pts_time - float(timestamp)), default=None)
        return None

    def build(self, candidates: Sequence[Any], *, limit: int) -> tuple[FrameCandidate, ...]:
        frames: list[FrameCandidate] = []
        seen: set[tuple[str, int]] = set()
        for candidate in candidates:
            metadata = dict(getattr(candidate, "metadata", {}) or {})
            video_id = metadata.get("video_id", getattr(candidate, "video_id", None))
            if not video_id:
                continue
            record = self._resolve_record(str(video_id), metadata)
            if record is None or record.image_path is None:
                continue
            key = (str(video_id), record.frame_idx)
            if key in seen:
                continue
            seen.add(key)
            frames.append(
                FrameCandidate(
                    candidate_id=f"C{len(frames) + 1:02d}",
                    video_id=str(video_id),
                    frame_idx=record.frame_idx,
                    timestamp=record.pts_time,
                    image_path=record.image_path,
                    retrieval_score=float(getattr(candidate, "score", 0.0)),
                    asr_text=str(metadata.get("asr_text", "")),
                    ocr_text=str(metadata.get("ocr_text", "")),
                )
            )
            if len(frames) >= limit:
                break
        if not frames:
            raise ValueError("No retrieval candidate could be resolved to a local keyframe image")
        return tuple(frames)


class QASolver:
    def __init__(
        self,
        verifier: QAVerifier,
        manifest: DatasetManifest,
        *,
        config: QAConfig | None = None,
    ) -> None:
        self.verifier = verifier
        self.config = config or QAConfig()
        self.builder = QACandidateBuilder(manifest)

    def solve(
        self,
        *,
        query_id: str,
        question: str,
        retrieval_context: str,
        candidates: Sequence[Any],
    ) -> QAVerification:
        """Make one grounded Gemini call for cheap/single-answer smoke tests."""

        frames = self.builder.build(candidates, limit=self.config.max_candidates)
        request = QAVerificationRequest(
            query_id=query_id,
            question=question,
            retrieval_context=retrieval_context,
            language=self.config.language,
            candidates=frames,
        )
        return self.verifier.verify(request)

    def solve_ranked(
        self,
        *,
        query_id: str,
        question: str,
        retrieval_context: str,
        candidates: Sequence[Any],
    ) -> list[RankedQAAnswer]:
        """Competition mode for retaining alternative ranked answers.

        This intentionally makes one Gemini call per disjoint batch. Every batch
        comes exclusively from candidates that local retrieval selected and the
        local manifest resolved to trusted frame paths and IDs.
        """

        frames = self.builder.build(candidates, limit=self.config.max_candidates)
        raw: list[tuple[QAVerification, float]] = []
        retrieval_scores = [frame.retrieval_score or 0.0 for frame in frames]
        low, high = min(retrieval_scores), max(retrieval_scores)
        normalized = {
            frame.candidate_id: (
                1.0 if high == low else ((frame.retrieval_score or 0.0) - low) / (high - low)
            )
            for frame in frames
        }
        for batch_number, start in enumerate(range(0, len(frames), self.config.batch_size), 1):
            batch = frames[start : start + self.config.batch_size]
            request = QAVerificationRequest(
                query_id=f"{query_id}-B{batch_number:02d}",
                question=question,
                retrieval_context=retrieval_context,
                language=self.config.language,
                candidates=batch,
            )
            result = self.verifier.verify(request)
            if not result.answerable or result.supporting_candidate_id is None:
                continue
            score = 0.55 * result.confidence + 0.45 * normalized[result.supporting_candidate_id]
            raw.append((result, score))
        seen: set[tuple[str | None, int | None, str | None]] = set()
        answers: list[RankedQAAnswer] = []
        for result, score in sorted(raw, key=lambda item: -item[1]):
            key = (
                result.supporting_video_id,
                result.supporting_frame_idx,
                result.normalized_answer,
            )
            if key in seen:
                continue
            seen.add(key)
            answers.append(RankedQAAnswer(result=result, score=score))
        return answers
