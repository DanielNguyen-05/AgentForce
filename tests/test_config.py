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
    assert config.gemini.max_output_tokens == 2048
    assert config.gemini.max_retry_output_tokens == 8192
    assert config.gemini.thinking_level == "minimal"


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


def test_gemini_output_and_thinking_settings_are_loaded(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[gemini]
max_output_tokens = 3072
max_retry_output_tokens = 12288
thinking_level = "low"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.gemini.max_output_tokens == 3072
    assert config.gemini.max_retry_output_tokens == 12288
    assert config.gemini.thinking_level == "low"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("timeout_seconds", "0", "timeout_seconds"),
        ("prompt_version", '\"\"', "prompt_version"),
        ("max_candidates", "65", "max_candidates"),
        ("max_output_tokens", "0", "max_output_tokens"),
        ("max_retry_output_tokens", "0", "max_retry_output_tokens"),
    ),
)
def test_invalid_gemini_runtime_settings_fail_fast(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(f"[gemini]\n{field} = {value}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_config(config_path)


def test_gemini_retry_output_budget_cannot_be_smaller_than_initial_budget(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[gemini]
max_output_tokens = 4096
max_retry_output_tokens = 2048
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="max_retry_output_tokens"):
        load_config(config_path)
