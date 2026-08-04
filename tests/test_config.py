from pathlib import Path

import pytest

from agentforce.config import load_config
from agentforce.errors import ConfigurationError


def test_load_default_config(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[paths]
dataset_root = "data"
artifacts_root = "out"
[windows]
length_seconds = 12.0
stride_seconds = 4.0
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    config = load_config(config_path)
    assert config.paths.dataset_root == (tmp_path / "data").resolve()
    assert config.paths.transcripts_dir == (tmp_path / "out" / "transcripts").resolve()
    assert config.windows.length_seconds == 12.0
    assert config.retrieval.top_keyframes == 500


def test_asr_hotwords_and_audio_settings_are_loaded(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[asr]
hotwords = [" Buôn Ma Thuột ", "xe đầu kéo"]
audio_sample_rate = 16000
audio_channels = 1
normalize_lufs = true
integrated_loudness = -16.0
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.asr.hotwords == ("Buôn Ma Thuột", "xe đầu kéo")
    assert config.asr.normalize_lufs is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("timeout_seconds", "0", "timeout_seconds"),
        ("prompt_version", '\"\"', "prompt_version"),
        ("max_candidates", "65", "max_candidates"),
    ),
)
def test_invalid_gemini_runtime_settings_fail_fast(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(f"[gemini]\n{field} = {value}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_config(config_path)
