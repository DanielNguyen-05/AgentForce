from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentforce.data.schemas import OCRItem
from agentforce.preprocessing.asr import ASRConfig, FasterWhisperAdapter
from agentforce.preprocessing.ocr import EasyOCREngine, OCRProcessor, normalize_ocr_text


class FakeWhisper:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def transcribe(self, path, **kwargs):
        del path
        self.kwargs = kwargs
        segment = SimpleNamespace(
            id=3,
            start=1.0,
            end=2.0,
            text=" xin chào ",
            avg_logprob=-0.1,
            no_speech_prob=0.01,
            words=[SimpleNamespace(word="xin", start=1.0, end=1.4, probability=0.9)],
        )
        info = SimpleNamespace(language="vi", language_probability=0.98, duration=3.0)
        return iter([segment]), info


class FakeOCR:
    name = "fake"

    def recognize(self, image_path):
        del image_path
        return [
            OCRItem("  ĐÀ   NẴNG ", "đà nẵng", 0.9),
            OCRItem("noise", "noise", 0.1),
        ]


class WarningEasyOCRReader:
    def readtext(self, image_path):
        del image_path
        warnings.warn(
            "'pin_memory' argument is set as true but no accelerator is found, "
            "then device pinned memory won't be used.",
            UserWarning,
        )
        warnings.warn(
            "'pin_memory' argument is set as true but not supported on MPS now, "
            "device pinned memory won't be used.",
            UserWarning,
        )
        warnings.warn("a useful EasyOCR warning", UserWarning)
        return [([[0, 0], [10, 0], [10, 5], [0, 5]], "Xin chào", 0.9)]


class BatchEasyOCRReader:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def readtext_batched(self, image_paths, *, batch_size):
        paths = list(image_paths)
        self.calls.append((paths, batch_size))
        return [
            [
                (
                    [[0, 0], [10, 0], [10, 5], [0, 5]],
                    f"text-{Path(path).stem}",
                    0.9,
                )
            ]
            for path in paths
        ]


def test_faster_whisper_adapter_accepts_injected_model(tmp_path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake")
    model = FakeWhisper()
    adapter = FasterWhisperAdapter(
        ASRConfig(model_name="fake", hotwords=("Buôn Ma Thuột", "xe đầu kéo")),
        model=model,
    )
    document = adapter.transcribe(audio, video_id="L01_V001")
    assert document.language == "vi"
    assert document.text == "xin chào"
    assert document.segments[0].words[0].confidence == 0.9
    assert document.segments[0].confidence is not None
    assert model.kwargs["hotwords"] == "Buôn Ma Thuột, xe đầu kéo"


def test_ocr_processor_filters_confidence_and_normalizes_unicode(tmp_path) -> None:
    image = tmp_path / "001.jpg"
    image.write_bytes(b"fake")
    result = OCRProcessor(FakeOCR(), min_confidence=0.3).process(image, "L01_V001_K000001")
    assert len(result.items) == 1
    assert result.normalized_text == "đà nẵng"
    assert normalize_ocr_text("  ĐÀ   NẴNG ") == "đà nẵng"


def test_easyocr_suppresses_only_repetitive_pin_memory_warnings() -> None:
    engine = EasyOCREngine(suppress_pin_memory_warnings=True)
    engine._reader = WarningEasyOCRReader()

    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        items = engine.recognize("frame.jpg")

    assert [item.text for item in items] == ["Xin chào"]
    assert [str(item.message) for item in observed] == ["a useful EasyOCR warning"]


def test_easyocr_can_expose_pin_memory_warnings_for_debugging() -> None:
    engine = EasyOCREngine(suppress_pin_memory_warnings=False)
    engine._reader = WarningEasyOCRReader()

    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        engine.recognize("frame.jpg")

    messages = [str(item.message) for item in observed]
    assert any("no accelerator" in message for message in messages)
    assert any("not supported on MPS" in message for message in messages)


def test_easyocr_batches_keyframes_without_changing_order_or_schema(tmp_path) -> None:
    reader = BatchEasyOCRReader()
    engine = EasyOCREngine(batch_size=2)
    engine._reader = reader
    processor = OCRProcessor(engine)
    paths = [tmp_path / name for name in ("003.jpg", "001.jpg", "002.jpg")]
    output = tmp_path / "ocr.jsonl"

    processor.process_many(
        ((f"L21_V001_K{number:06d}", path) for number, path in enumerate(paths, 1)),
        output,
    )

    assert reader.calls == [
        ([str(paths[0]), str(paths[1])], 2),
        ([str(paths[2])], 2),
    ]
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["keyframe_uid"] for row in rows] == [
        "L21_V001_K000001",
        "L21_V001_K000002",
        "L21_V001_K000003",
    ]
    assert [row["full_text"] for row in rows] == ["text-003", "text-001", "text-002"]
    assert all(row["engine"] == "easyocr" for row in rows)


def test_easyocr_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        EasyOCREngine(batch_size=0)
