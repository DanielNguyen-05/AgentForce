from __future__ import annotations

import json
from pathlib import Path

import pytest

import convert_phowhisper as converter_module
from convert_phowhisper import (
    CONVERSION_MANIFEST,
    REQUIRED_MODEL_FILES,
    ConversionRequest,
    PhoWhisperConversionError,
    convert_phowhisper,
)


class FakeConverter:
    def __init__(self, *, complete: bool = True, marker: bytes = b"new") -> None:
        self.complete = complete
        self.marker = marker
        self.calls: list[tuple[str, str, bool]] = []

    def convert(self, output_dir: str, *, quantization: str, force: bool) -> None:
        self.calls.append((output_dir, quantization, force))
        destination = Path(output_dir)
        destination.mkdir(parents=True)
        filenames = REQUIRED_MODEL_FILES if self.complete else ("model.bin",)
        for filename in filenames:
            content = b"{}" if filename.endswith(".json") else self.marker
            (destination / filename).write_bytes(content)


def _write_valid_model(path: Path, marker: bytes = b"old") -> None:
    path.mkdir(parents=True)
    for filename in REQUIRED_MODEL_FILES:
        content = b"{}" if filename.endswith(".json") else marker
        (path / filename).write_bytes(content)


def test_conversion_uses_staging_validates_and_writes_manifest(tmp_path) -> None:
    output = tmp_path / "phowhisper-ct2"
    fake = FakeConverter()
    request = ConversionRequest(
        source_model="vinai/PhoWhisper-small",
        output_dir=output,
        quantization="int8",
        revision="test-revision",
    )

    result = convert_phowhisper(request, converter_factory=lambda _: fake)

    assert result["status"] == "converted"
    assert (output / "model.bin").read_bytes() == b"new"
    manifest = json.loads((output / CONVERSION_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["source_model"] == "vinai/PhoWhisper-small"
    assert manifest["source_revision"] == "test-revision"
    assert manifest["quantization"] == "int8"
    assert set(manifest["files"]) == set(REQUIRED_MODEL_FILES)
    assert fake.calls[0][1:] == ("int8", False)
    assert not list(tmp_path.glob(".phowhisper-ct2.partial-*"))


def test_failed_forced_conversion_preserves_existing_model(tmp_path) -> None:
    output = tmp_path / "phowhisper-ct2"
    _write_valid_model(output)
    fake = FakeConverter(complete=False)

    with pytest.raises(PhoWhisperConversionError, match="missing"):
        convert_phowhisper(
            ConversionRequest(output_dir=output, force=True),
            converter_factory=lambda _: fake,
        )

    assert (output / "model.bin").read_bytes() == b"old"
    assert not list(tmp_path.glob(".phowhisper-ct2.partial-*"))
    assert not list(tmp_path.glob(".phowhisper-ct2.backup-*"))


def test_successful_forced_conversion_replaces_existing_model(tmp_path) -> None:
    output = tmp_path / "phowhisper-ct2"
    _write_valid_model(output)
    fake = FakeConverter(marker=b"replacement")

    result = convert_phowhisper(
        ConversionRequest(output_dir=output, force=True),
        converter_factory=lambda _: fake,
    )

    assert result["status"] == "converted"
    assert (output / "model.bin").read_bytes() == b"replacement"
    assert not list(tmp_path.glob(".phowhisper-ct2.backup-*"))


def test_valid_existing_model_is_reused_without_loading_converter(tmp_path) -> None:
    output = tmp_path / "phowhisper-ct2"
    _write_valid_model(output)

    def unexpected_factory(_: ConversionRequest):
        raise AssertionError("converter should not be loaded")

    result = convert_phowhisper(
        ConversionRequest(output_dir=output), converter_factory=unexpected_factory
    )

    assert result["status"] == "skipped"
    assert (output / "model.bin").read_bytes() == b"old"


def test_relative_output_is_resolved_from_project_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(converter_module, "PROJECT_ROOT", tmp_path)
    fake = FakeConverter()

    result = convert_phowhisper(
        ConversionRequest(output_dir=Path("models/test-model")),
        converter_factory=lambda _: fake,
    )

    assert Path(str(result["output_dir"])) == (tmp_path / "models/test-model").resolve()


def test_converter_refuses_to_replace_project_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(converter_module, "PROJECT_ROOT", tmp_path)

    with pytest.raises(PhoWhisperConversionError, match="unsafe"):
        convert_phowhisper(
            ConversionRequest(output_dir=Path("."), force=True),
            converter_factory=lambda _: FakeConverter(),
        )
