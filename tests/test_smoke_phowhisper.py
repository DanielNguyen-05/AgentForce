from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from agentforce.data.schemas import (
    TranscriptDocument,
    TranscriptSegment,
    TranscriptWord,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
SCRIPT_PATH = SCRIPTS_ROOT / "smoke_phowhisper.py"
spec = importlib.util.spec_from_file_location("test_script_smoke_phowhisper", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_validate_output_path_only_allows_smoke_json(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    accepted = outputs / "smoke" / "phowhisper" / "clip.json"
    assert (
        smoke.validate_output_path(accepted, outputs_root=outputs, overwrite=False)
        == accepted.resolve()
    )

    with pytest.raises(ValueError, match="must stay under"):
        smoke.validate_output_path(
            tmp_path / "artifacts" / "transcripts" / "L21_V001.json",
            outputs_root=outputs,
            overwrite=False,
        )
    with pytest.raises(ValueError, match="must be a .json"):
        smoke.validate_output_path(
            outputs / "smoke" / "clip.txt",
            outputs_root=outputs,
            overwrite=False,
        )


def test_validate_output_path_requires_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "smoke" / "clip.json"
    output.parent.mkdir(parents=True)
    output.write_text("do not replace\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        smoke.validate_output_path(
            output,
            outputs_root=tmp_path / "outputs",
            overwrite=False,
        )
    assert output.read_text(encoding="utf-8") == "do not replace\n"


@pytest.mark.parametrize(
    ("start", "duration", "message"),
    ((-0.1, 10.0, "cannot be negative"), (0.0, 0.0, "must be > 0"), (0.0, 60.1, "<= 60")),
)
def test_validate_clip_rejects_unsafe_ranges(start: float, duration: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        smoke.validate_clip(
            start_seconds=start,
            duration_seconds=duration,
            video_duration_seconds=100.0,
        )


def test_smoke_payload_keeps_relative_and_absolute_timestamps(tmp_path: Path) -> None:
    document = TranscriptDocument(
        video_id="L21_V001",
        language="vi",
        language_probability=0.99,
        duration_seconds=12.0,
        model_name="fake-model",
        segments=[
            TranscriptSegment(
                segment_id=0,
                start=0.5,
                end=2.0,
                text="xin chào",
                words=[TranscriptWord(text="xin", start=0.5, end=0.9, confidence=0.8)],
            )
        ],
    )
    payload = smoke.build_smoke_payload(
        document,
        source_video=tmp_path / "video.mp4",
        model_reference="models/fake",
        device="cpu",
        compute_type="int8",
        requested_language="vi",
        start_seconds=4.0,
        requested_duration_seconds=12.0,
        extraction_seconds=0.1,
        transcription_seconds=1.2,
        total_seconds=1.3,
    )

    assert payload["kind"] == "phowhisper_smoke"
    assert payload["text"] == "xin chào"
    segment = payload["segments"][0]
    assert segment["start"] == 0.5
    assert segment["absolute_start"] == 4.5
    assert segment["words"][0]["absolute_end"] == 4.9
    assert payload["clip"]["absolute_end_seconds"] == 16.0


def test_extract_audio_clip_uses_bounded_ffmpeg_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "clip.wav"
    observed: list[str] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        observed.extend(command)
        output.write_bytes(b"RIFF-test")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    smoke.extract_audio_clip(
        tmp_path / "video.mp4",
        output,
        start_seconds=4.0,
        duration_seconds=12.0,
    )

    assert observed[observed.index("-ss") + 1] == "4.000000"
    assert observed[observed.index("-t") + 1] == "12.000000"
    assert observed[observed.index("-ar") + 1] == "16000"
    assert observed[-1] == str(output)
