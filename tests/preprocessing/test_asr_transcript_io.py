from __future__ import annotations

import json

import pytest

from agentforce.data.schemas import TranscriptDocument, TranscriptSegment, TranscriptWord
from agentforce.preprocessing.asr import (
    TranscriptSchemaError,
    migrate_transcript,
    read_transcript,
    write_transcript,
)


def _legacy_payload() -> dict[str, object]:
    return {
        "source_video": "L21_V001.mp4",
        "generated_at": "2026-08-04T19:36:00",
        "pipeline_config": {
            "asr_model": "faster-whisper-models/phowhisper-large-ct2",
            "device": "cpu",
            "compute_type": "int8",
        },
        "audio_info": {
            "duration_sec": 12.0,
            "detected_language": "vi",
            "language_probability": 1,
        },
        "segments": [
            {
                "segment_id": 1,
                "time_range": {"start": 1.0, "end": 3.0},
                "audio_metadata": {
                    "raw_text": "xin chào",
                    "clean_text": "Xin chào.",
                    "confidence": 0.95,
                    "no_speech_prob": 0.01,
                    "words": [
                        {"word": "xin", "start": 1.0, "end": 1.4, "probability": 0.9},
                        {
                            "word": "chào",
                            "start": 1.4,
                            "end": 2.0,
                            "probability": 0.92,
                        },
                    ],
                },
            }
        ],
    }


def test_canonical_transcript_round_trip(tmp_path) -> None:
    source = tmp_path / "canonical.json"
    expected = TranscriptDocument(
        video_id="L21_V001",
        language="vi",
        language_probability=0.99,
        duration_seconds=4.0,
        segments=[
            TranscriptSegment(
                segment_id=0,
                start=1.0,
                end=2.0,
                text="xin chào",
                words=[TranscriptWord("xin", 1.0, 1.4, 0.9)],
                avg_logprob=-0.1,
                no_speech_prob=0.01,
                confidence=0.9,
            )
        ],
        model_name="vinai/PhoWhisper-large",
        source_path="dataset/videos/L21/L21_V001.mp4",
    )

    write_transcript(expected, source)

    assert read_transcript(source) == expected


def test_legacy_phowhisper_schema_is_mapped_to_canonical(tmp_path) -> None:
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps(_legacy_payload()), encoding="utf-8")

    document = read_transcript(source)

    assert document.video_id == "L21_V001"
    assert document.model_name == "faster-whisper-models/phowhisper-large-ct2"
    assert document.language == "vi"
    assert document.duration_seconds == 12.0
    assert document.segments[0].text == "Xin chào."
    assert document.segments[0].avg_logprob is None
    assert document.segments[0].words[1] == TranscriptWord("chào", 1.4, 2.0, 0.92)


def test_legacy_transcript_can_be_atomically_rewritten_in_place(tmp_path) -> None:
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps(_legacy_payload()), encoding="utf-8")

    document = read_transcript(source, rewrite_legacy=True)
    rewritten = json.loads(source.read_text(encoding="utf-8"))

    assert rewritten["video_id"] == "L21_V001"
    assert "source_video" not in rewritten
    assert read_transcript(source) == document
    assert not source.with_name(source.name + ".tmp").exists()


def test_migration_refuses_to_overwrite_a_distinct_destination(tmp_path) -> None:
    source = tmp_path / "legacy.json"
    destination = tmp_path / "canonical.json"
    source.write_text(json.dumps(_legacy_payload()), encoding="utf-8")
    destination.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        migrate_transcript(source, destination)

    assert destination.read_text(encoding="utf-8") == "keep"


def test_migration_writes_a_distinct_canonical_destination(tmp_path) -> None:
    source = tmp_path / "legacy.json"
    destination = tmp_path / "canonical.json"
    source.write_text(json.dumps(_legacy_payload()), encoding="utf-8")

    document = migrate_transcript(source, destination)

    assert read_transcript(destination) == document
    assert "source_video" in json.loads(source.read_text(encoding="utf-8"))


def test_strict_reader_rejects_word_outside_segment(tmp_path) -> None:
    source = tmp_path / "invalid.json"
    payload = _legacy_payload()
    payload["segments"][0]["audio_metadata"]["words"][0]["start"] = 0.0  # type: ignore[index]
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TranscriptSchemaError, match="outside its segment"):
        read_transcript(source)


def test_strict_reader_rejects_unknown_schema(tmp_path) -> None:
    source = tmp_path / "invalid.json"
    source.write_text('{"segments": []}', encoding="utf-8")

    with pytest.raises(TranscriptSchemaError, match="Unsupported transcript schema"):
        read_transcript(source)
