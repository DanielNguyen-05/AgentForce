from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agentforce.preprocessing.audio import extract_audio, probe_media


def test_extract_audio_uses_argument_list_and_atomic_output(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.mp4"
    output = tmp_path / "audio.wav"
    source.write_bytes(b"video")
    observed: list[str] = []

    def fake_run(command, **kwargs):
        del kwargs
        observed.extend(command)
        Path(command[-1]).write_bytes(b"wave")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("agentforce.preprocessing.audio.subprocess.run", fake_run)
    assert extract_audio(source, output).read_bytes() == b"wave"
    assert observed[0] == "ffmpeg"
    assert observed[observed.index("-ar") + 1] == "16000"
    assert "-vn" in observed


def test_probe_media_extracts_video_and_audio_properties(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "duration": "2.0",
                "avg_frame_rate": "30000/1001",
                "nb_frames": "60",
                "width": 1920,
                "height": 1080,
            },
            {"codec_type": "audio"},
        ],
        "format": {"duration": "2.1"},
    }

    def fake_run(command, **kwargs):
        del command, kwargs
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(payload))

    monkeypatch.setattr("agentforce.preprocessing.audio.subprocess.run", fake_run)
    probe = probe_media(source)
    assert probe.has_audio
    assert probe.frame_count == 60
    assert probe.width == 1920
    assert probe.fps is not None and round(probe.fps, 2) == 29.97
