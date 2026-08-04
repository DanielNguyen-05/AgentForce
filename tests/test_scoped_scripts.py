from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agentforce.data.schemas import DatasetManifest, VideoRecord
from agentforce.data.scope import scope_hash

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

VIDEO_IDS = ("L21_V001", "L21_V002", "L21_V003")
SCOPED_SCRIPT_NAMES = (
    "validate_dataset",
    "export_timelines",
    "normalize_objects",
    "run_ocr",
    "build_windows",
    "build_visual_indexes",
    "build_text_indexes",
)


def _load_script(name: str):
    path = SCRIPTS_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scoped_project(tmp_path: Path) -> tuple[Path, DatasetManifest]:
    config_path = tmp_path / "scope.toml"
    config_path.write_text(
        """
[paths]
dataset_root = "dataset"
artifacts_root = "artifacts"
outputs_root = "outputs"

[scope]
video_ids = ["L21_V001", "L21_V002", "L21_V003"]

[embeddings]
clip_model = "test-clip"
clip_pretrained = "test-weights"
text_model = "test-text"
device = "cpu"
batch_size = 8
""",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = DatasetManifest(
        dataset_root=str(dataset),
        videos=[
            VideoRecord(video_id=video_id, collection="L21", duration_seconds=10.0)
            for video_id in VIDEO_IDS
        ],
    )
    manifest_path = tmp_path / "artifacts" / "manifests" / "dataset.json"
    manifest.write_json(manifest_path)
    return config_path, manifest


@pytest.mark.parametrize("script_name", SCOPED_SCRIPT_NAMES)
def test_scoped_scripts_accept_plural_video_ids(script_name: str) -> None:
    parser = _load_script(script_name).build_parser()
    args = parser.parse_args(["--video-ids", "L21_V001", "L21_V002"])
    assert args.video_ids == ["L21_V001", "L21_V002"]


@pytest.mark.parametrize(
    "script_name",
    (
        "export_timelines",
        "normalize_objects",
        "run_ocr",
        "build_windows",
        "build_visual_indexes",
        "build_text_indexes",
    ),
)
def test_subset_cannot_write_canonical_artifacts(
    script_name: str,
    scoped_project: tuple[Path, DatasetManifest],
) -> None:
    config_path, _ = scoped_project
    module = _load_script(script_name)
    with pytest.raises(ValueError, match="does not match configured scope"):
        module.main(
            ["--config", str(config_path), "--video-ids", "L21_V001"]
        )


def test_validate_dataset_cannot_escape_or_replace_canonical_with_subset(
    scoped_project: tuple[Path, DatasetManifest], monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, manifest = scoped_project
    module = _load_script("validate_dataset")
    monkeypatch.setattr(module, "build_manifest", lambda *_args, **_kwargs: manifest)

    with pytest.raises(ValueError, match="does not match configured scope"):
        module.main(
            ["--config", str(config_path), "--video-ids", "L21_V001"]
        )
    with pytest.raises(ValueError, match="outside the configured scope"):
        module.main(
            ["--config", str(config_path), "--video-ids", "L22_V001"]
        )


@pytest.mark.parametrize(
    ("script_name", "extra_args"),
    (
        ("export_timelines", ["--limit", "1"]),
        ("normalize_objects", ["--limit", "1"]),
        ("normalize_objects", ["--frame-limit", "1"]),
        ("run_ocr", ["--limit", "1"]),
        ("run_ocr", ["--frame-limit", "1"]),
        ("build_windows", ["--limit", "1"]),
        ("build_visual_indexes", ["--level", "window", "--limit", "1"]),
        ("build_text_indexes", ["--limit", "1"]),
    ),
)
def test_partial_options_require_noncanonical_output(
    script_name: str,
    extra_args: list[str],
    scoped_project: tuple[Path, DatasetManifest],
) -> None:
    config_path, _ = scoped_project
    module = _load_script(script_name)
    with pytest.raises(ValueError, match="non-canonical"):
        module.main(["--config", str(config_path), *extra_args])


def test_build_windows_reads_phowhisper_from_outputs_root(
    scoped_project: tuple[Path, DatasetManifest],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path, _ = scoped_project
    transcript = tmp_path / "outputs" / "transcripts" / "L21_V001.json"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("{}\n", encoding="utf-8")
    module = _load_script("build_windows")
    observed: list[Path] = []
    monkeypatch.setattr(module, "timeline_from_manifest", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(module, "build_temporal_windows", lambda *_args, **_kwargs: [])

    def read_transcript(path: Path):
        observed.append(path)
        return SimpleNamespace(segments=[])

    monkeypatch.setattr(module, "read_transcript", read_transcript)
    output = tmp_path / "smoke" / "windows.jsonl"
    assert (
        module.main(
            [
                "--config",
                str(config_path),
                "--video-ids",
                "L21_V001",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert observed == [transcript]
    persisted = json.loads(output.with_suffix(".manifest.json").read_text())
    assert persisted["video_ids"] == ["L21_V001"]
    assert persisted["scope_hash"] == scope_hash(["L21_V001"])


def _write_visual_inputs(root: Path) -> list[dict[str, object]]:
    features = root / "dataset" / "clip-features-32"
    mappings = root / "dataset" / "map-keyframes"
    windows_path = root / "artifacts" / "windows" / "temporal_windows.jsonl"
    features.mkdir(parents=True)
    mappings.mkdir(parents=True)
    windows_path.parent.mkdir(parents=True)
    windows: list[dict[str, object]] = []
    for position, video_id in enumerate(VIDEO_IDS):
        np.save(
            features / f"{video_id}.npy",
            np.asarray([[1.0 + position, 1.0]], dtype=np.float16),
        )
        (mappings / f"{video_id}.csv").write_text(
            "n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n", encoding="utf-8"
        )
        windows.append(
            {
                "window_id": f"{video_id}_W000000",
                "video_id": video_id,
                "keyframe_uids": [f"{video_id}_K000001"],
                "start_time": 0.0,
                "end_time": 10.0,
                "object_labels": ["person"],
            }
        )
    windows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in windows), encoding="utf-8"
    )
    return windows


def test_visual_and_text_index_manifests_record_exact_scope_and_encoders(
    scoped_project: tuple[Path, DatasetManifest], tmp_path: Path
) -> None:
    config_path, _ = scoped_project
    _write_visual_inputs(tmp_path)

    visual_script = _load_script("build_visual_indexes")
    assert visual_script.main(["--config", str(config_path)]) == 0
    index_root = tmp_path / "artifacts" / "indexes"
    expected_visual_encoder = {"model": "test-clip", "pretrained": "test-weights"}
    for name in ("visual_keyframes", "visual_windows"):
        manifest = json.loads((index_root / f"{name}.manifest.json").read_text())
        assert manifest["video_ids"] == list(VIDEO_IDS)
        assert manifest["scope_hash"] == scope_hash(VIDEO_IDS)
        assert manifest["expected_encoder"] == expected_visual_encoder

    text_script = _load_script("build_text_indexes")
    assert (
        text_script.main(
            [
                "--config",
                str(config_path),
                "--encoder",
                "hashing",
                "--hash-dimension",
                "8",
                "--fields",
                "object_labels",
            ]
        )
        == 0
    )
    text_manifest = json.loads(
        (index_root / "objects_windows.manifest.json").read_text()
    )
    assert text_manifest["video_ids"] == list(VIDEO_IDS)
    assert text_manifest["scope_hash"] == scope_hash(VIDEO_IDS)
    assert text_manifest["expected_encoder"] == {
        "class": "HashingTextEncoder",
        "model": None,
        "dimension": 8,
    }
