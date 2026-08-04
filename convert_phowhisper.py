"""Convert an official VinAI PhoWhisper checkpoint to CTranslate2.

This is a one-time, project-local model preparation step.  Transcription is
performed by ``scripts/transcribe_videos.py`` after this converter succeeds.

Example for the current Apple Silicon/CPU setup::

    python convert_phowhisper.py \
        --source-model vinai/PhoWhisper-large \
        --output-dir models/phowhisper-large-ct2 \
        --quantization int8

The Hugging Face checkpoint is downloaded by Transformers on the first run.
Conversion happens in a temporary sibling directory, so an interrupted run
does not leave a half-written model at the requested output path.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_MODEL = "vinai/PhoWhisper-large"
DEFAULT_OUTPUT_DIR = Path("models/phowhisper-large-ct2")
DEFAULT_COPY_FILES = ("tokenizer.json", "preprocessor_config.json")
REQUIRED_MODEL_FILES = (
    "model.bin",
    "config.json",
    "tokenizer.json",
    "preprocessor_config.json",
)
JSON_MODEL_FILES = ("config.json", "tokenizer.json", "preprocessor_config.json")
QUANTIZATION_TYPES = (
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "int16",
    "float16",
    "bfloat16",
    "float32",
)
CONVERSION_MANIFEST = "conversion_manifest.json"


class PhoWhisperConversionError(RuntimeError):
    """Raised when conversion cannot safely produce a usable local model."""


@dataclass(frozen=True, slots=True)
class ConversionRequest:
    source_model: str = DEFAULT_SOURCE_MODEL
    output_dir: Path = DEFAULT_OUTPUT_DIR
    quantization: str = "int8"
    revision: str | None = None
    low_cpu_mem_usage: bool = False
    force: bool = False

    def __post_init__(self) -> None:
        if not self.source_model.strip():
            raise ValueError("source_model cannot be empty")
        if self.quantization not in QUANTIZATION_TYPES:
            raise ValueError(f"Unsupported quantization: {self.quantization}")


ConverterFactory = Callable[[ConversionRequest], Any]


def project_path(path: str | Path) -> Path:
    """Resolve user-facing relative paths from the repository root."""

    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def validate_converted_model(model_dir: str | Path) -> dict[str, int]:
    """Validate the minimum files Faster-Whisper needs from a local CT2 model."""

    root = Path(model_dir)
    if not root.is_dir():
        raise PhoWhisperConversionError(f"Converted model directory does not exist: {root}")
    sizes: dict[str, int] = {}
    missing: list[str] = []
    empty: list[str] = []
    for filename in REQUIRED_MODEL_FILES:
        path = root / filename
        if not path.is_file():
            missing.append(filename)
            continue
        size = path.stat().st_size
        if size <= 0:
            empty.append(filename)
        sizes[filename] = size
    if missing or empty:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if empty:
            details.append("empty: " + ", ".join(empty))
        raise PhoWhisperConversionError(
            f"Invalid CTranslate2 model at {root} ({'; '.join(details)})"
        )
    invalid_json: list[str] = []
    for filename in JSON_MODEL_FILES:
        try:
            payload = json.loads((root / filename).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            invalid_json.append(filename)
            continue
        if not isinstance(payload, dict):
            invalid_json.append(filename)
    if invalid_json:
        raise PhoWhisperConversionError(
            f"Invalid CTranslate2 model at {root} (malformed JSON: "
            f"{', '.join(invalid_json)})"
        )
    return sizes


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("ctranslate2", "transformers", "torch", "faster-whisper"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _write_manifest(
    staging_dir: Path,
    request: ConversionRequest,
    file_sizes: dict[str, int],
) -> None:
    payload = {
        "schema_version": "1.0",
        "source_model": request.source_model,
        "source_revision": request.revision,
        "quantization": request.quantization,
        "copy_files": list(DEFAULT_COPY_FILES),
        "low_cpu_mem_usage": request.low_cpu_mem_usage,
        "created_at": datetime.now(UTC).isoformat(),
        "package_versions": _package_versions(),
        "files": {
            filename: {"size_bytes": size}
            for filename, size in sorted(file_sizes.items())
        },
    }
    destination = staging_dir / CONVERSION_MANIFEST
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _default_converter_factory(request: ConversionRequest) -> Any:
    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError as exc:
        raise PhoWhisperConversionError(
            "PhoWhisper conversion requires CTranslate2, Transformers and PyTorch. "
            "Install the project PhoWhisper conversion dependencies first."
        ) from exc
    return TransformersConverter(
        request.source_model,
        copy_files=list(DEFAULT_COPY_FILES),
        revision=request.revision,
        low_cpu_mem_usage=request.low_cpu_mem_usage,
        trust_remote_code=False,
    )


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def convert_phowhisper(
    request: ConversionRequest,
    *,
    converter_factory: ConverterFactory | None = None,
) -> dict[str, object]:
    """Convert one model without exposing a partial output directory.

    A valid existing output is reused unless ``force`` is set.  With ``force``,
    the old directory remains in place until the replacement has been fully
    converted and validated.
    """

    output_dir = project_path(request.output_dir)
    dangerous_targets = {
        Path("/").resolve(),
        Path.home().resolve(),
        PROJECT_ROOT.resolve(),
        PROJECT_ROOT.resolve().parent,
    }
    if output_dir in dangerous_targets:
        raise PhoWhisperConversionError(
            f"Refusing unsafe model output directory: {output_dir}"
        )
    if output_dir.exists() and not request.force:
        sizes = validate_converted_model(output_dir)
        return {
            "status": "skipped",
            "output_dir": str(output_dir),
            "reason": "a valid converted model already exists; use --force to replace it",
            "files": sizes,
        }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(f".{output_dir.name}.partial-{uuid4().hex}")
    backup_dir: Path | None = None
    factory = converter_factory or _default_converter_factory

    try:
        converter = factory(request)
        converter.convert(
            str(staging_dir),
            quantization=request.quantization,
            force=False,
        )
        sizes = validate_converted_model(staging_dir)
        _write_manifest(staging_dir, request, sizes)

        if output_dir.exists():
            backup_dir = output_dir.with_name(f".{output_dir.name}.backup-{uuid4().hex}")
            output_dir.replace(backup_dir)
        staging_dir.replace(output_dir)
        if backup_dir is not None:
            _remove_path(backup_dir)
            backup_dir = None
    except Exception:
        _remove_path(staging_dir)
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            backup_dir.replace(output_dir)
            backup_dir = None
        raise
    finally:
        if backup_dir is not None and backup_dir.exists():
            _remove_path(backup_dir)

    return {
        "status": "converted",
        "source_model": request.source_model,
        "quantization": request.quantization,
        "output_dir": str(output_dir),
        "manifest": str(output_dir / CONVERSION_MANIFEST),
        "files": sizes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--quantization", choices=QUANTIZATION_TYPES, default="int8")
    parser.add_argument("--revision", help="Optional Hugging Face revision/commit")
    parser.add_argument(
        "--low-cpu-mem-usage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ask Transformers to reduce peak CPU memory (requires Accelerate)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Safely replace an existing converted model after validating the new one",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[RUN] {Path(__file__).resolve()}", file=sys.stderr)
    request = ConversionRequest(
        source_model=args.source_model,
        output_dir=Path(args.output_dir),
        quantization=args.quantization,
        revision=args.revision,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        force=args.force,
    )
    try:
        result = convert_phowhisper(request)
    except (PhoWhisperConversionError, OSError, RuntimeError, ValueError) as exc:
        print(f"PhoWhisper conversion failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
