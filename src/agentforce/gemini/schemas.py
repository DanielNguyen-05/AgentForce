"""Validated domain records for multi-frame Gemini Q&A."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    stripped = value.strip()
    return stripped or None


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class FrameCandidate:
    """One locally retrieved frame that Gemini may select as evidence."""

    candidate_id: str
    video_id: str
    frame_idx: int
    timestamp: float
    image_path: Path | str
    retrieval_score: float | None = None
    asr_text: str = ""
    ocr_text: str = ""
    caption_text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _required_text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "video_id", _required_text(self.video_id, "video_id"))
        if isinstance(self.frame_idx, bool) or not isinstance(self.frame_idx, int) or self.frame_idx < 0:
            raise ValueError("frame_idx must be a non-negative integer")
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, float)):
            raise ValueError("timestamp must be numeric")
        if float(self.timestamp) < 0:
            raise ValueError("timestamp must be non-negative")
        object.__setattr__(self, "timestamp", float(self.timestamp))
        image_path = Path(self.image_path)
        if not str(image_path):
            raise ValueError("image_path must not be empty")
        object.__setattr__(self, "image_path", image_path)
        if self.retrieval_score is not None:
            object.__setattr__(self, "retrieval_score", float(self.retrieval_score))

    def metadata_dict(self) -> dict[str, Any]:
        """Return metadata safe for logs/cache keys (without image bytes)."""

        return {
            "candidate_id": self.candidate_id,
            "video_id": self.video_id,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "image_path": str(self.image_path),
            "retrieval_score": self.retrieval_score,
            "asr_text": self.asr_text,
            "ocr_text": self.ocr_text,
            "caption_text": self.caption_text,
        }


@dataclass(frozen=True, slots=True)
class QAVerificationRequest:
    """A question and the small candidate set produced by local retrieval."""

    query_id: str
    question: str
    candidates: tuple[FrameCandidate, ...] | Sequence[FrameCandidate]
    retrieval_context: str = ""
    language: str = "vi"

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _required_text(self.query_id, "query_id"))
        object.__setattr__(self, "question", _required_text(self.question, "question"))
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("At least one frame candidate is required")
        if len(candidates) > 64:
            raise ValueError("At most 64 frame candidates may be sent in one request")
        if not all(isinstance(item, FrameCandidate) for item in candidates):
            raise TypeError("candidates must contain FrameCandidate values")
        identifiers = [item.candidate_id for item in candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate_id values must be unique within a request")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "language", _required_text(self.language, "language"))


@dataclass(frozen=True, slots=True)
class QAVerification:
    """Semantically validated answer returned by Gemini.

    ``supporting_video_id`` and ``supporting_frame_idx`` are resolved from the
    local candidate registry by :class:`QAVerifier`; callers never have to
    trust model-generated identifiers.
    """

    answerable: bool
    answer: str | None
    normalized_answer: str | None
    answer_type: str
    supporting_candidate_id: str | None
    confidence: float
    evidence: tuple[str, ...] = field(default_factory=tuple)
    uncertainty_reason: str | None = None
    supporting_video_id: str | None = None
    supporting_frame_idx: int | None = None
    cache_hit: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.answerable, bool):
            raise ValueError("answerable must be a boolean")
        object.__setattr__(self, "answer", _optional_text(self.answer, "answer"))
        object.__setattr__(
            self,
            "normalized_answer",
            _optional_text(self.normalized_answer, "normalized_answer"),
        )
        object.__setattr__(self, "answer_type", _required_text(self.answer_type, "answer_type"))
        object.__setattr__(
            self,
            "supporting_candidate_id",
            _optional_text(self.supporting_candidate_id, "supporting_candidate_id"),
        )
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        if isinstance(self.evidence, str) or not isinstance(self.evidence, Sequence):
            raise ValueError("evidence must be an array of strings")
        evidence = tuple(_required_text(item, "evidence item") for item in self.evidence)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(
            self,
            "uncertainty_reason",
            _optional_text(self.uncertainty_reason, "uncertainty_reason"),
        )
        object.__setattr__(
            self,
            "supporting_video_id",
            _optional_text(self.supporting_video_id, "supporting_video_id"),
        )
        if self.supporting_frame_idx is not None:
            frame_idx = _integer(self.supporting_frame_idx, "supporting_frame_idx")
            if frame_idx < 0:
                raise ValueError("supporting_frame_idx must be non-negative")
        if self.answerable:
            if self.answer is None or self.normalized_answer is None:
                raise ValueError("An answerable result requires answer and normalized_answer")
            if self.supporting_candidate_id is None:
                raise ValueError("An answerable result requires supporting_candidate_id")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_local_support(self, candidate: FrameCandidate, *, cache_hit: bool) -> "QAVerification":
        return QAVerification(
            answerable=self.answerable,
            answer=self.answer,
            normalized_answer=self.normalized_answer,
            answer_type=self.answer_type,
            supporting_candidate_id=self.supporting_candidate_id,
            confidence=self.confidence,
            evidence=self.evidence,
            uncertainty_reason=self.uncertainty_reason,
            supporting_video_id=candidate.video_id,
            supporting_frame_idx=candidate.frame_idx,
            cache_hit=cache_hit,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, cache_hit: bool = False) -> "QAVerification":
        if not isinstance(value, Mapping):
            raise ValueError("Gemini response must be a JSON object")
        required = {
            "answerable",
            "answer",
            "normalized_answer",
            "answer_type",
            "supporting_candidate_id",
            "confidence",
            "evidence",
            "uncertainty_reason",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"Gemini response is missing fields: {', '.join(missing)}")
        return cls(
            answerable=value["answerable"],
            answer=value["answer"],
            normalized_answer=value["normalized_answer"],
            answer_type=value["answer_type"],
            supporting_candidate_id=value["supporting_candidate_id"],
            confidence=value["confidence"],
            evidence=value["evidence"],
            uncertainty_reason=value["uncertainty_reason"],
            cache_hit=cache_hit,
        )

    @staticmethod
    def json_schema() -> dict[str, Any]:
        """Schema sent to Gemini structured output mode.

        Keep it deliberately shallow: Gemini supports a useful subset of JSON
        Schema and can reject unnecessarily complex definitions.
        """

        nullable_string = {"type": ["string", "null"]}
        return {
            "type": "object",
            "properties": {
                "answerable": {
                    "type": "boolean",
                    "description": "True only when supplied visual evidence directly answers the question.",
                },
                "answer": {
                    **nullable_string,
                    "description": "A concise answer in the requested language, or null.",
                },
                "normalized_answer": {
                    **nullable_string,
                    "description": "Canonical short form used for evaluation, or null.",
                },
                "answer_type": {
                    "type": "string",
                    "description": "Semantic type such as color, count, person, object, action, text, or unknown.",
                },
                "supporting_candidate_id": {
                    **nullable_string,
                    "description": "ID of the single supplied frame with the strongest direct evidence, or null.",
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "At most three short observations grounded in supplied inputs.",
                },
                "uncertainty_reason": {
                    **nullable_string,
                    "description": "Why evidence is insufficient or ambiguous, otherwise null.",
                },
            },
            "required": [
                "answerable",
                "answer",
                "normalized_answer",
                "answer_type",
                "supporting_candidate_id",
                "confidence",
                "evidence",
                "uncertainty_reason",
            ],
            "additionalProperties": False,
        }
