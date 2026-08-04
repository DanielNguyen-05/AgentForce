"""Read-only integrity checks for canonical experiment artifacts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agentforce.config import AppConfig
from agentforce.utils.io import canonical_jsonl_sha256, file_sha256

from .schemas import DatasetManifest
from .scope import normalize_video_ids, scope_hash


@dataclass(frozen=True, slots=True)
class ArtifactIssue:
    severity: str
    code: str
    message: str
    path: str | None = None
    video_id: str | None = None

    def __post_init__(self) -> None:
        if self.severity not in {"warning", "error"}:
            raise ValueError(f"Unsupported artifact issue severity: {self.severity}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArtifactValidationReport:
    config_path: str | None
    scope_video_ids: tuple[str, ...]
    strict: bool
    require_ocr: bool
    checks: dict[str, Any] = field(default_factory=dict)
    issues: list[ArtifactIssue] = field(default_factory=list)

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        *,
        path: Path | str | None = None,
        video_id: str | None = None,
    ) -> None:
        self.issues.append(
            ArtifactIssue(
                severity=severity,
                code=code,
                message=message,
                path=str(path) if path is not None else None,
                video_id=video_id,
            )
        )

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "config": self.config_path,
            "strict": self.strict,
            "require_ocr": self.require_ocr,
            "scope": {
                "video_ids": list(self.scope_video_ids),
                "video_count": len(self.scope_video_ids),
                "scope_hash": scope_hash(self.scope_video_ids),
            },
            "ok": self.error_count == 0,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "checks": self.checks,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class JsonlSummary:
    row_count: int
    video_counts: Mapping[str, int]
    valid: bool

    @property
    def video_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.video_counts))


def _format_ids(video_ids: Sequence[str], *, limit: int = 12) -> str:
    values = sorted(set(video_ids))
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f" (+{len(values) - limit} more)"
    return f"{shown}{suffix}"


def _compact_ids(video_ids: Sequence[str], *, limit: int = 20) -> dict[str, Any]:
    values = sorted(set(video_ids))
    return {
        "count": len(values),
        "values": values[:limit],
        "truncated": max(0, len(values) - limit),
    }


def _load_json_object(
    path: Path,
    report: ArtifactValidationReport,
    *,
    missing_code: str,
    invalid_code: str,
    severity: str = "error",
) -> dict[str, Any] | None:
    if not path.is_file():
        report.add(severity, missing_code, f"Missing required JSON file: {path}", path=path)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        report.add(severity, invalid_code, f"Cannot read JSON object: {exc}", path=path)
        return None
    if not isinstance(payload, dict):
        report.add(severity, invalid_code, "JSON root must be an object", path=path)
        return None
    return payload


def _compare_scope(
    actual_ids: Sequence[str],
    expected_ids: Sequence[str],
    report: ArtifactValidationReport,
    *,
    code: str,
    label: str,
    path: Path,
    severity: str = "error",
) -> bool:
    raw = [str(value).strip().upper() for value in actual_ids]
    duplicates = sorted(video_id for video_id, count in Counter(raw).items() if count > 1)
    actual = set(raw)
    expected = set(expected_ids)
    if duplicates or actual != expected:
        details: list[str] = []
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            details.append(f"missing={_format_ids(missing)}")
        if extra:
            details.append(f"extra={_format_ids(extra)}")
        if duplicates:
            details.append(f"duplicates={_format_ids(duplicates)}")
        report.add(
            severity,
            code,
            f"{label} does not exactly match configured scope ({', '.join(details)})",
            path=path,
        )
        return False
    return True


def _row_video_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("video_id")
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    uid = row.get("keyframe_uid")
    if isinstance(uid, str) and "_K" in uid.upper():
        return uid.upper().rsplit("_K", 1)[0]
    return None


def _read_jsonl(
    path: Path,
    report: ArtifactValidationReport,
    *,
    label: str,
    severity: str = "error",
    video_id_getter: Callable[[Mapping[str, Any]], str | None] = _row_video_id,
) -> JsonlSummary | None:
    if not path.is_file():
        report.add(
            severity,
            f"missing_{label}",
            f"Missing {label} JSONL file: {path}",
            path=path,
        )
        return None
    counts: Counter[str] = Counter()
    row_count = 0
    valid = True
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row_count += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    valid = False
                    report.add(
                        severity,
                        f"invalid_{label}_jsonl",
                        f"Invalid JSON at line {line_number}: {exc}",
                        path=path,
                    )
                    continue
                if not isinstance(row, dict):
                    valid = False
                    report.add(
                        severity,
                        f"invalid_{label}_row",
                        f"Line {line_number} must contain a JSON object",
                        path=path,
                    )
                    continue
                video_id = video_id_getter(row)
                if video_id is None:
                    valid = False
                    report.add(
                        severity,
                        f"missing_{label}_video_id",
                        f"Cannot resolve video ID at line {line_number}",
                        path=path,
                    )
                    continue
                counts[video_id] += 1
    except (OSError, UnicodeError) as exc:
        report.add(
            severity,
            f"unreadable_{label}",
            f"Cannot read {label} JSONL: {exc}",
            path=path,
        )
        return None
    return JsonlSummary(row_count=row_count, video_counts=dict(counts), valid=valid)


def _validate_manifest(
    config: AppConfig,
    expected_scope: tuple[str, ...],
    report: ArtifactValidationReport,
) -> tuple[DatasetManifest | None, dict[str, int]]:
    path = config.paths.artifacts_root / "manifests" / "dataset.json"
    try:
        manifest = DatasetManifest.read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        report.add("error", "invalid_dataset_manifest", f"Cannot load manifest: {exc}", path=path)
        report.checks["dataset_manifest"] = {"path": str(path), "valid": False}
        return None, {}

    actual_ids = [video.video_id for video in manifest.videos]
    scope_valid = _compare_scope(
        actual_ids,
        expected_scope,
        report,
        code="dataset_manifest_scope_mismatch",
        label="Dataset manifest",
        path=path,
    )
    expected_counts: dict[str, int] = {}
    for video in manifest.videos:
        if video.video_id not in expected_scope:
            continue
        if video.keyframe_count is None or video.keyframe_count < 0:
            report.add(
                "error",
                "manifest_missing_keyframe_count",
                "Manifest has no valid keyframe_count",
                path=path,
                video_id=video.video_id,
            )
        else:
            expected_counts[video.video_id] = video.keyframe_count
    report.checks["dataset_manifest"] = {
        "path": str(path),
        "valid": scope_valid and len(expected_counts) == len(expected_scope),
        "video_ids": _compact_ids(actual_ids),
        "keyframe_total": sum(expected_counts.values()),
    }
    return manifest, expected_counts


def _validate_transcripts(
    config: AppConfig,
    expected_scope: tuple[str, ...],
    report: ArtifactValidationReport,
) -> None:
    directory = config.paths.outputs_root / "transcripts"
    valid_ids: list[str] = []
    if not directory.is_dir():
        report.add(
            "error",
            "missing_transcript_directory",
            f"Missing transcript directory: {directory}",
            path=directory,
        )
        report.checks["transcripts"] = {"path": str(directory), "valid_video_ids": []}
        return

    expected = set(expected_scope)
    outside_paths = [
        path for path in sorted(directory.glob("*.json")) if path.stem.upper() not in expected
    ]
    if outside_paths:
        report.add(
            "error",
            "transcript_outside_scope",
            "Transcript files are outside configured scope: "
            + _format_ids([path.stem.upper() for path in outside_paths]),
            path=directory,
        )
    for video_id in expected_scope:
        path = directory / f"{video_id}.json"
        payload = _load_json_object(
            path,
            report,
            missing_code="missing_transcript",
            invalid_code="invalid_transcript",
        )
        if payload is None:
            continue
        declared_ids: list[str] = []
        declared = payload.get("video_id")
        if isinstance(declared, str) and declared.strip():
            declared_ids.append(declared.strip().upper())
        source_video = payload.get("source_video")
        if isinstance(source_video, str) and source_video.strip():
            declared_ids.append(Path(source_video).stem.upper())
        if not declared_ids:
            report.add(
                "error",
                "transcript_missing_video_id",
                "Transcript must declare video_id or source_video",
                path=path,
                video_id=video_id,
            )
            continue
        if any(declared_id != video_id for declared_id in declared_ids):
            report.add(
                "error",
                "transcript_video_id_mismatch",
                f"Transcript declares {declared_ids}, expected {video_id}",
                path=path,
                video_id=video_id,
            )
            continue
        valid_ids.append(video_id)
    report.checks["transcripts"] = {
        "path": str(directory),
        "valid_video_ids": valid_ids,
        "expected_count": len(expected_scope),
    }


def _validate_timeline(
    config: AppConfig,
    expected_scope: tuple[str, ...],
    expected_counts: Mapping[str, int],
    report: ArtifactValidationReport,
) -> None:
    path = config.paths.artifacts_root / "timelines" / "keyframes.jsonl"
    summary = _read_jsonl(path, report, label="timeline")
    if summary is None:
        report.checks["timeline"] = {"path": str(path), "valid": False}
        return
    scope_valid = _compare_scope(
        summary.video_ids,
        expected_scope,
        report,
        code="timeline_scope_mismatch",
        label="Canonical timeline rows",
        path=path,
    )
    for video_id, expected_count in expected_counts.items():
        actual_count = summary.video_counts.get(video_id, 0)
        if actual_count != expected_count:
            report.add(
                "error",
                "timeline_keyframe_count_mismatch",
                f"Timeline has {actual_count} rows, expected {expected_count}",
                path=path,
                video_id=video_id,
            )
    expected_total = sum(expected_counts.values())
    if expected_counts and summary.row_count != expected_total:
        report.add(
            "error",
            "timeline_total_mismatch",
            f"Timeline has {summary.row_count} rows, expected {expected_total}",
            path=path,
        )
    report.checks["timeline"] = {
        "path": str(path),
        "valid": summary.valid and scope_valid and summary.row_count == expected_total,
        "row_count": summary.row_count,
        "expected_row_count": expected_total,
        "video_ids": _compact_ids(summary.video_ids),
    }


def _validate_per_video_jsonl(
    directory: Path,
    *,
    kind: str,
    expected_scope: tuple[str, ...],
    expected_counts: Mapping[str, int],
    report: ArtifactValidationReport,
    severity: str,
) -> None:
    expected = set(expected_scope)
    if directory.is_dir():
        outside_paths = [
            path
            for path in sorted(directory.glob("*.jsonl"))
            if path.stem.upper() not in expected
        ]
        if outside_paths:
            report.add(
                severity,
                f"{kind}_artifact_outside_scope",
                f"{kind.upper()} artifacts are outside configured scope: "
                + _format_ids([path.stem.upper() for path in outside_paths]),
                path=directory,
            )
    details: dict[str, Any] = {}
    for video_id in expected_scope:
        path = directory / f"{video_id}.jsonl"
        summary = _read_jsonl(path, report, label=kind, severity=severity)
        if summary is None:
            details[video_id] = {"path": str(path), "valid": False}
            continue
        actual_count = summary.row_count
        expected_count = expected_counts.get(video_id)
        if expected_count is not None and actual_count != expected_count:
            report.add(
                severity,
                f"{kind}_row_count_mismatch",
                f"{kind.upper()} has {actual_count} rows, expected {expected_count}",
                path=path,
                video_id=video_id,
            )
        wrong_ids = sorted(set(summary.video_ids) - {video_id})
        if wrong_ids:
            report.add(
                severity,
                f"{kind}_video_id_mismatch",
                f"{kind.upper()} rows resolve to unexpected video IDs: {wrong_ids}",
                path=path,
                video_id=video_id,
            )
        details[video_id] = {
            "path": str(path),
            "valid": (
                summary.valid
                and not wrong_ids
                and (expected_count is None or actual_count == expected_count)
            ),
            "row_count": actual_count,
            "expected_row_count": expected_count,
        }
    report.checks[kind] = details


def _validate_windows(
    config: AppConfig,
    expected_scope: tuple[str, ...],
    report: ArtifactValidationReport,
) -> str | None:
    path = config.paths.artifacts_root / "windows" / "temporal_windows.jsonl"
    manifest_path = path.with_suffix(".manifest.json")
    payload = _load_json_object(
        manifest_path,
        report,
        missing_code="missing_window_manifest",
        invalid_code="invalid_window_manifest",
    )
    summary = _read_jsonl(path, report, label="temporal_windows")
    check: dict[str, Any] = {
        "path": str(path),
        "manifest": str(manifest_path),
        "valid": payload is not None and summary is not None,
    }
    canonical_hash: str | None = None
    if summary is not None:
        check["row_count"] = summary.row_count
        check["video_ids"] = _compact_ids(summary.video_ids)
        if not summary.valid:
            check["valid"] = False
        if not _compare_scope(
            summary.video_ids,
            expected_scope,
            report,
            code="window_row_scope_mismatch",
            label="Temporal-window rows",
            path=path,
        ):
            check["valid"] = False
        try:
            canonical_hash = canonical_jsonl_sha256(path, exclude_fields=("metadata_text",))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            report.add(
                "error",
                "window_canonical_hash_failed",
                f"Cannot hash canonical window records: {exc}",
                path=path,
            )
            check["valid"] = False
    if payload is not None:
        manifest_ids = payload.get("video_ids")
        if not isinstance(manifest_ids, list) or not all(
            isinstance(item, str) for item in manifest_ids
        ):
            report.add(
                "error",
                "window_manifest_missing_scope",
                "Window manifest video_ids must be an array of strings",
                path=manifest_path,
            )
            check["valid"] = False
        elif not _compare_scope(
            manifest_ids,
            expected_scope,
            report,
            code="window_manifest_scope_mismatch",
            label="Window manifest",
            path=manifest_path,
        ):
            check["valid"] = False
        expected_scope_hash = scope_hash(expected_scope)
        if payload.get("scope_hash") != expected_scope_hash:
            report.add(
                "error",
                "window_scope_hash_mismatch",
                f"Window scope_hash must equal {expected_scope_hash}",
                path=manifest_path,
            )
            check["valid"] = False
        if path.is_file():
            try:
                actual_file_hash = file_sha256(path)
            except OSError as exc:
                report.add(
                    "error",
                    "window_file_hash_failed",
                    f"Cannot hash temporal-window file: {exc}",
                    path=path,
                )
                check["valid"] = False
            else:
                if payload.get("sha256") != actual_file_hash:
                    report.add(
                        "error",
                        "window_file_hash_mismatch",
                        "Window manifest sha256 does not match temporal_windows.jsonl",
                        path=manifest_path,
                    )
                    check["valid"] = False
        if canonical_hash is not None and payload.get("canonical_records_hash") != canonical_hash:
            report.add(
                "error",
                "window_canonical_hash_mismatch",
                "Window canonical_records_hash does not match current records",
                path=manifest_path,
            )
            check["valid"] = False
        if payload.get("video_count") != len(expected_scope):
            report.add(
                "error",
                "window_video_count_mismatch",
                f"Window manifest video_count must equal {len(expected_scope)}",
                path=manifest_path,
            )
            check["valid"] = False
    check["canonical_records_hash"] = canonical_hash
    report.checks["temporal_windows"] = check
    return canonical_hash


def _resolve_index_member(
    root: Path,
    raw_path: object,
    report: ArtifactValidationReport,
    *,
    manifest_path: Path,
    kind: str,
) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        report.add(
            "error",
            f"index_missing_{kind}_path",
            f"Index manifest must declare {kind}_path",
            path=manifest_path,
        )
        return None
    candidate = Path(raw_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        report.add(
            "error",
            f"index_{kind}_outside_root",
            f"Index {kind} path escapes the index directory",
            path=manifest_path,
        )
        return None
    return resolved


def _validate_index(
    manifest_path: Path,
    *,
    expected_scope: tuple[str, ...],
    expected_scope_hash: str,
    window_hash: str | None,
    report: ArtifactValidationReport,
) -> None:
    payload = _load_json_object(
        manifest_path,
        report,
        missing_code="missing_index_manifest",
        invalid_code="invalid_index_manifest",
    )
    name = manifest_path.name.removesuffix(".manifest.json")
    check: dict[str, Any] = {"manifest": str(manifest_path), "valid": payload is not None}
    if payload is None:
        report.checks.setdefault("indexes", {})[name] = check
        return

    declared_name = payload.get("name")
    if declared_name != name:
        report.add(
            "error",
            "index_name_mismatch",
            f"Index manifest declares name={declared_name!r}, expected {name!r}",
            path=manifest_path,
        )
        check["valid"] = False
    manifest_ids = payload.get("video_ids")
    if not isinstance(manifest_ids, list) or not all(isinstance(item, str) for item in manifest_ids):
        report.add(
            "error",
            "index_manifest_missing_scope",
            "Index manifest video_ids must be an array of strings",
            path=manifest_path,
        )
        check["valid"] = False
    elif not _compare_scope(
        manifest_ids,
        expected_scope,
        report,
        code="index_manifest_scope_mismatch",
        label=f"Index {name}",
        path=manifest_path,
    ):
        check["valid"] = False
    if payload.get("scope_hash") != expected_scope_hash:
        report.add(
            "error",
            "index_scope_hash_mismatch",
            f"Index scope_hash must equal {expected_scope_hash}",
            path=manifest_path,
        )
        check["valid"] = False
    if name.endswith("_windows") and (
        window_hash is None or payload.get("window_records_hash") != window_hash
    ):
        report.add(
            "error",
            "index_window_hash_mismatch",
            "Window index hash does not match canonical temporal windows",
            path=manifest_path,
        )
        check["valid"] = False

    root = manifest_path.parent
    vectors_path = _resolve_index_member(
        root,
        payload.get("vectors_path"),
        report,
        manifest_path=manifest_path,
        kind="vectors",
    )
    metadata_path = _resolve_index_member(
        root,
        payload.get("metadata_path"),
        report,
        manifest_path=manifest_path,
        kind="metadata",
    )
    if vectors_path is None or metadata_path is None:
        check["valid"] = False
    expected_rows = payload.get("row_count")
    expected_dimension = payload.get("dimension")
    expected_dtype = payload.get("dtype")
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows < 0:
        report.add("error", "index_invalid_row_count", "Invalid manifest row_count", path=manifest_path)
        expected_rows = None
        check["valid"] = False
    if (
        isinstance(expected_dimension, bool)
        or not isinstance(expected_dimension, int)
        or expected_dimension < 1
    ):
        report.add(
            "error", "index_invalid_dimension", "Invalid manifest dimension", path=manifest_path
        )
        expected_dimension = None
        check["valid"] = False
    if not isinstance(expected_dtype, str) or expected_dtype not in {"float16", "float32"}:
        report.add(
            "error",
            "index_invalid_dtype",
            "Index dtype must be float16 or float32",
            path=manifest_path,
        )
        expected_dtype = None
        check["valid"] = False

    if vectors_path is not None:
        if not vectors_path.is_file():
            report.add(
                "error", "missing_index_vectors", "Missing index vector matrix", path=vectors_path
            )
            check["valid"] = False
        else:
            try:
                matrix = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError, EOFError) as exc:
                report.add(
                    "error",
                    "invalid_index_vectors",
                    f"Cannot read index vector matrix: {exc}",
                    path=vectors_path,
                )
                check["valid"] = False
            else:
                check["vector_shape"] = list(matrix.shape)
                check["vector_dtype"] = str(matrix.dtype)
                if matrix.ndim != 2:
                    report.add(
                        "error",
                        "index_vectors_not_2d",
                        f"Index vectors must be 2D, got shape {matrix.shape}",
                        path=vectors_path,
                    )
                    check["valid"] = False
                else:
                    if expected_rows is not None and matrix.shape[0] != expected_rows:
                        report.add(
                            "error",
                            "index_vector_row_count_mismatch",
                            f"Vector rows={matrix.shape[0]}, manifest rows={expected_rows}",
                            path=vectors_path,
                        )
                        check["valid"] = False
                    if expected_dimension is not None and matrix.shape[1] != expected_dimension:
                        report.add(
                            "error",
                            "index_vector_dimension_mismatch",
                            f"Vector dimension={matrix.shape[1]}, manifest dimension={expected_dimension}",
                            path=vectors_path,
                        )
                        check["valid"] = False
                if expected_dtype is not None and str(matrix.dtype) != expected_dtype:
                    report.add(
                        "error",
                        "index_vector_dtype_mismatch",
                        f"Vector dtype={matrix.dtype}, manifest dtype={expected_dtype}",
                        path=vectors_path,
                    )
                    check["valid"] = False

    if metadata_path is not None:
        summary = _read_jsonl(metadata_path, report, label="index_metadata")
        if summary is None:
            check["valid"] = False
        else:
            check["metadata_rows"] = summary.row_count
            check["metadata_video_ids"] = _compact_ids(summary.video_ids)
            if expected_rows is not None and summary.row_count != expected_rows:
                report.add(
                    "error",
                    "index_metadata_row_count_mismatch",
                    f"Metadata rows={summary.row_count}, manifest rows={expected_rows}",
                    path=metadata_path,
                )
                check["valid"] = False
            outside = sorted(set(summary.video_ids) - set(expected_scope))
            if outside:
                report.add(
                    "error",
                    "index_metadata_outside_scope",
                    "Index metadata contains out-of-scope video IDs: "
                    + _format_ids(outside),
                    path=metadata_path,
                )
                check["valid"] = False
            if not summary.valid:
                check["valid"] = False
    report.checks.setdefault("indexes", {})[name] = check


def _validate_indexes(
    config: AppConfig,
    expected_scope: tuple[str, ...],
    window_hash: str | None,
    report: ArtifactValidationReport,
) -> None:
    directory = config.paths.artifacts_root / "indexes"
    manifests = sorted(directory.glob("*.manifest.json")) if directory.is_dir() else []
    available = {path.name.removesuffix(".manifest.json") for path in manifests}
    if report.strict:
        required = {"visual_keyframes", "visual_windows", "asr_windows", "objects_windows"}
        for name in sorted(required - available):
            report.add(
                "error",
                "missing_required_index",
                f"Strict validation requires index {name}",
                path=directory / f"{name}.manifest.json",
            )
    expected_hash = scope_hash(expected_scope)
    for manifest_path in manifests:
        _validate_index(
            manifest_path,
            expected_scope=expected_scope,
            expected_scope_hash=expected_hash,
            window_hash=window_hash,
            report=report,
        )
    for path in sorted(directory.glob("*")) if directory.is_dir() else []:
        if not path.is_file() or path.suffix not in {".npy", ".jsonl"}:
            continue
        if not (directory / f"{path.stem}.manifest.json").is_file():
            report.add(
                "error",
                "orphan_index_file",
                "Index data file has no matching manifest",
                path=path,
            )


def _validate_no_captions(config: AppConfig, report: ArtifactValidationReport) -> None:
    root = config.paths.artifacts_root
    if not root.exists():
        return
    paths = sorted(
        path
        for path in root.rglob("*")
        if any("caption" in part.casefold() for part in path.relative_to(root).parts)
    )
    if paths:
        shown = [str(path) for path in paths[:20]]
        suffix = "" if len(paths) <= 20 else f" (+{len(paths) - 20} more)"
        report.add(
            "error",
            "caption_artifacts_forbidden",
            f"Caption artifacts/indexes are no longer supported: {shown}{suffix}",
            path=root,
        )
    report.checks["caption_artifacts"] = {
        "valid": not paths,
        "matches": [str(path) for path in paths],
    }


def validate_artifacts(
    config: AppConfig,
    *,
    strict: bool = False,
    require_ocr: bool = False,
) -> ArtifactValidationReport:
    """Validate the configured artifact set without modifying any artifact."""

    expected_scope = normalize_video_ids(config.scope.video_ids)
    report = ArtifactValidationReport(
        config_path=str(config.source_path) if config.source_path is not None else None,
        scope_video_ids=expected_scope,
        strict=strict,
        require_ocr=require_ocr,
    )
    if not expected_scope:
        report.add(
            "error",
            "empty_configured_scope",
            "Artifact validation requires explicit [scope].video_ids",
            path=config.source_path,
        )
        return report

    _, expected_counts = _validate_manifest(config, expected_scope, report)
    _validate_transcripts(config, expected_scope, report)
    _validate_timeline(config, expected_scope, expected_counts, report)
    _validate_per_video_jsonl(
        config.paths.artifacts_root / "objects",
        kind="objects",
        expected_scope=expected_scope,
        expected_counts=expected_counts,
        report=report,
        severity="error",
    )
    _validate_per_video_jsonl(
        config.paths.artifacts_root / "ocr",
        kind="ocr",
        expected_scope=expected_scope,
        expected_counts=expected_counts,
        report=report,
        severity="error" if require_ocr else "warning",
    )
    window_hash = _validate_windows(config, expected_scope, report)
    _validate_indexes(config, expected_scope, window_hash, report)
    _validate_no_captions(config, report)
    return report


__all__ = [
    "ArtifactIssue",
    "ArtifactValidationReport",
    "validate_artifacts",
]
