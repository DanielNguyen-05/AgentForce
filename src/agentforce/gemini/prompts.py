"""Prompt construction kept separate for easy versioning and debugging."""

from __future__ import annotations

from .schemas import QAVerificationRequest


PROMPT_VERSION = "qa-multiframe-v1"

SYSTEM_INSTRUCTION = """You are a visual question-answering verifier.
Use only the supplied candidate frames and their local ASR/OCR/caption context.
Never use outside knowledge or invent content hidden outside a frame.
If evidence is insufficient, set answerable to false.
Choose exactly one candidate ID that most directly supports an answer.
Answer concisely in the requested language.
For counts, do not infer occluded people or objects.
For visible text, preserve the text shown in the evidence.
"""


def build_qa_prompt(request: QAVerificationRequest) -> str:
    """Build the text portion of a multimodal request.

    The transport inserts each image immediately after its corresponding
    ``FRAME <candidate_id>`` marker. The output JSON shape is intentionally not
    duplicated here because it is supplied through structured-output config.
    """

    lines = [
        f"Prompt version: {PROMPT_VERSION}",
        f"Query ID: {request.query_id}",
        f"Answer language: {request.language}",
        f"Question: {request.question}",
    ]
    if request.retrieval_context.strip():
        lines.append(f"Retrieval description (for locating the scene only): {request.retrieval_context.strip()}")
    lines.extend(
        [
            "",
            "Evaluate every supplied frame before answering.",
            "Candidate metadata follows; images are attached with matching candidate IDs.",
        ]
    )
    for candidate in request.candidates:
        score = "unknown" if candidate.retrieval_score is None else f"{candidate.retrieval_score:.6f}"
        lines.extend(
            [
                "",
                (
                    f"CANDIDATE {candidate.candidate_id}: video={candidate.video_id}; "
                    f"frame_idx={candidate.frame_idx}; timestamp={candidate.timestamp:.3f}s; "
                    f"retrieval_score={score}"
                ),
                f"ASR: {candidate.asr_text.strip() or '[none]'}",
                f"OCR: {candidate.ocr_text.strip() or '[none]'}",
                f"Caption hint: {candidate.caption_text.strip() or '[none]'}",
            ]
        )
    lines.extend(
        [
            "",
            "Return answerable=false when no frame provides direct evidence.",
            "The supporting_candidate_id must exactly match one candidate ID listed above.",
        ]
    )
    return "\n".join(lines)
