from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


def _load_script():
    path = SCRIPTS_ROOT / "run_kis_batch.py"
    spec = importlib.util.spec_from_file_location("test_script_run_kis_batch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve postponed annotations through the defining module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_discover_kis_queries_filters_task_and_uses_natural_order(
    tmp_path: Path,
) -> None:
    module = _load_script()
    input_dir = tmp_path / "queries"
    input_dir.mkdir()
    for name in (
        "query-p1-10-kis.txt",
        "query-p1-2-kis.txt",
        "query-p1-1-kis.txt",
        "query-p1-3-qa.txt",
        "query-p1-4-trake.txt",
        "notes.md",
    ):
        (input_dir / name).write_text("query", encoding="utf-8")
    (input_dir / "nested-kis.txt").mkdir()

    discovered = module.discover_kis_queries(input_dir)

    assert [path.name for path in discovered] == [
        "query-p1-1-kis.txt",
        "query-p1-2-kis.txt",
        "query-p1-10-kis.txt",
    ]


def test_read_query_accepts_utf8_bom_and_rejects_empty_text(tmp_path: Path) -> None:
    module = _load_script()
    query_path = tmp_path / "query-p1-1-kis.txt"
    query_path.write_text("\ufeff  Cảnh người đang chạy.\n", encoding="utf-8")

    assert module.read_query(query_path) == "Cảnh người đang chạy."

    empty_path = tmp_path / "query-p1-2-kis.txt"
    empty_path.write_text(" \n\t", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        module.read_query(empty_path)


def test_load_retrieval_texts_validates_complete_exact_mapping(
    tmp_path: Path,
) -> None:
    module = _load_script()
    query_paths = [
        tmp_path / "query-p1-1-kis.txt",
        tmp_path / "query-p1-2-kis.txt",
    ]
    mapping_path = tmp_path / "retrieval.json"
    mapping_path.write_text(
        '\ufeff{"query-p1-1-kis": "  red   bus ", "query-p1-2-kis": "blue train"}',
        encoding="utf-8",
    )

    assert module.load_retrieval_texts(mapping_path, query_paths) == {
        "query-p1-1-kis": "red bus",
        "query-p1-2-kis": "blue train",
    }

    mapping_path.write_text('{"query-p1-1-kis": "red bus"}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing query IDs"):
        module.load_retrieval_texts(mapping_path, query_paths)

    mapping_path.write_text(
        '{"query-p1-1-kis": "red bus", "query-p1-2-kis": "blue train", "stale-kis": "stale"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown query IDs"):
        module.load_retrieval_texts(mapping_path, query_paths)


def test_load_reviewed_candidates_validates_and_preserves_order(tmp_path: Path) -> None:
    module = _load_script()
    query_paths = [tmp_path / "query-p1-1-kis.txt"]
    mapping_path = tmp_path / "reviewed.json"
    mapping_path.write_text(
        json.dumps(
            {
                "query-p1-1-kis": [
                    {
                        "video_id": "L21_V003",
                        "frame_id": 1739,
                        "note": "  dam in rain  ",
                    },
                    {"video_id": "L21_V003", "frame_id": 1600},
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = module.load_reviewed_candidates(mapping_path, query_paths)

    assert loaded["query-p1-1-kis"] == (
        module.ReviewedCandidate("L21_V003", 1739, "dam in rain"),
        module.ReviewedCandidate("L21_V003", 1600, ""),
    )

    mapping_path.write_text(
        '{"unknown-kis": [{"video_id": "L21_V003", "frame_id": 1}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown query IDs"):
        module.load_reviewed_candidates(mapping_path, query_paths)

    mapping_path.write_text(
        '{"query-p1-1-kis": [{"video_id": "bad.mp4", "frame_id": true}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid video_id"):
        module.load_reviewed_candidates(mapping_path, query_paths)


def test_validate_reviewed_candidates_checks_video_and_raw_frame_bounds(
    tmp_path: Path,
) -> None:
    module = _load_script()
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(
        json.dumps(
            {
                "videos": [
                    {"video_id": "L21_V001", "frame_count": 100},
                    {"video_id": "L21_V002", "frame_count": 200},
                ]
            }
        ),
        encoding="utf-8",
    )

    module.validate_reviewed_candidates_against_dataset(
        {"query-p1-1-kis": (module.ReviewedCandidate("L21_V001", 99),)},
        manifest_path,
    )

    with pytest.raises(ValueError, match="unknown video"):
        module.validate_reviewed_candidates_against_dataset(
            {"query-p1-1-kis": (module.ReviewedCandidate("L21_V999", 1),)},
            manifest_path,
        )
    with pytest.raises(ValueError, match="outside L21_V001"):
        module.validate_reviewed_candidates_against_dataset(
            {"query-p1-1-kis": (module.ReviewedCandidate("L21_V001", 100),)},
            manifest_path,
        )


def test_apply_reviewed_candidates_prepends_deduplicates_and_limits() -> None:
    module = _load_script()
    answers = [
        SimpleNamespace(
            video_id="L21_V001",
            frame_idx=10,
            score=0.8,
            candidate_id="automatic-1",
            timestamp=0.4,
            evidence={"visual": 0.8},
        ),
        SimpleNamespace(
            video_id="L21_V002",
            frame_idx=20,
            score=0.7,
            candidate_id="automatic-2",
            timestamp=0.8,
            evidence={},
        ),
    ]
    reviewed = (
        module.ReviewedCandidate("L21_V003", 30, "verified scene"),
        module.ReviewedCandidate("L21_V001", 10, "verified action"),
    )

    ranked = module.apply_reviewed_candidates(answers, reviewed, limit=3)

    assert [(item.video_id, item.frame_idx) for item in ranked] == [
        ("L21_V003", 30),
        ("L21_V001", 10),
        ("L21_V002", 20),
    ]
    assert ranked[0].evidence == {
        "human_reviewed": True,
        "review_note": "verified scene",
    }
    assert ranked[1].candidate_id == "automatic-1"
    assert ranked[1].timestamp == 0.4
    assert ranked[1].evidence == {
        "visual": 0.8,
        "human_reviewed": True,
        "review_note": "verified action",
    }


def test_run_batch_uses_retrieval_override_but_preserves_original_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    input_dir = tmp_path / "queries"
    input_dir.mkdir()
    original = "Một chiếc xe buýt màu đỏ."
    retrieval = "A red bus."
    (input_dir / "query-p1-1-kis.txt").write_text(original, encoding="utf-8")
    mapping_path = tmp_path / "retrieval.json"
    mapping_path.write_text(json.dumps({"query-p1-1-kis": retrieval}), encoding="utf-8")
    reviewed_path = tmp_path / "reviewed.json"
    reviewed_path.write_text(
        json.dumps(
            {
                "query-p1-1-kis": [
                    {
                        "video_id": "L21_V002",
                        "frame_id": 84,
                        "note": "manually verified",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    config = SimpleNamespace(
        retrieval=SimpleNamespace(
            top_keyframes=10,
            rrf_k=60,
            duplicate_seconds=2.0,
        )
    )
    monkeypatch.setattr(module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        module,
        "load_search_fields",
        lambda _config: [SimpleNamespace(name="visual", weight=1.0)],
    )
    monkeypatch.setattr(module, "build_text_searchers", lambda _fields: {})

    class FakeSolver:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def solve(self, text: str):
            assert text == retrieval
            return [
                SimpleNamespace(
                    video_id="L21_V001",
                    frame_idx=42,
                    score=0.9,
                    candidate_id="L21_V001_K000001",
                    timestamp=1.68,
                    evidence={},
                )
            ]

    monkeypatch.setattr(module, "KISSolver", FakeSolver)

    module.run_batch(
        input_dir,
        tmp_path / "debug",
        tmp_path / "submission",
        tmp_path / "config.toml",
        retrieval_texts_path=mapping_path,
        reviewed_candidates_path=reviewed_path,
    )

    payload = json.loads((tmp_path / "debug/query-p1-1-kis.json").read_text(encoding="utf-8"))
    assert payload["query"] == original
    assert payload["retrieval_text"] == retrieval
    assert payload["predictions"][0]["video_id"] == "L21_V002"
    assert payload["predictions"][0]["frame_ids"] == [84]
    assert payload["predictions"][0]["evidence"] == {
        "human_reviewed": True,
        "review_note": "manually verified",
    }
    assert (
        (tmp_path / "submission/query-1-kis.csv")
        .read_text(encoding="utf-8")
        .startswith("L21_V002,84\nL21_V001,42\n")
    )


def test_run_batch_loads_runtime_once_and_writes_one_result_per_input_stem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    input_dir = tmp_path / "queries"
    output_dir = tmp_path / "debug"
    submission_dir = tmp_path / "submission"
    input_dir.mkdir()
    output_dir.mkdir()
    submission_dir.mkdir()
    stale_json = output_dir / "query-p1-999-kis.json"
    stale_csv = submission_dir / "query-999-kis.csv"
    stale_json.write_text("{}\n", encoding="utf-8")
    stale_csv.write_text("L99_V999,1\n", encoding="utf-8")
    duplicate_query = "Cảnh một người mặc áo đỏ bước vào cửa hàng."
    unique_query = "Cảnh xe buýt dừng cạnh lề đường."
    (input_dir / "query-p1-10-kis.txt").write_text(duplicate_query, encoding="utf-8")
    (input_dir / "query-p1-2-kis.txt").write_text(unique_query, encoding="utf-8")
    (input_dir / "query-p1-1-kis.txt").write_text(duplicate_query, encoding="utf-8")
    (input_dir / "query-p1-3-qa.txt").write_text("ignored", encoding="utf-8")

    calls: dict[str, object] = {
        "load_config": 0,
        "load_search_fields": 0,
        "build_text_searchers": 0,
        "build_dense_refiner": 0,
        "solver_init": 0,
        "solve": [],
    }
    fields = [
        SimpleNamespace(name="visual", weight=1.0),
        SimpleNamespace(name="asr", weight=0.8),
    ]
    config = SimpleNamespace(
        retrieval=SimpleNamespace(
            top_keyframes=500,
            rrf_k=60,
            duplicate_seconds=2.0,
        )
    )
    searchers = {"visual": object(), "asr": object()}
    expected_refiner = object()

    def fake_load_config(path: Path):
        assert path == tmp_path / "config.toml"
        calls["load_config"] = int(calls["load_config"]) + 1
        return config

    def fake_load_search_fields(received_config):
        assert received_config is config
        calls["load_search_fields"] = int(calls["load_search_fields"]) + 1
        return fields

    def fake_build_text_searchers(received_fields):
        assert received_fields is fields
        calls["build_text_searchers"] = int(calls["build_text_searchers"]) + 1
        return searchers

    def fake_build_dense_refiner(received_config):
        assert received_config is config
        calls["build_dense_refiner"] = int(calls["build_dense_refiner"]) + 1
        return expected_refiner

    class FakeSolver:
        def __init__(
            self,
            received_searchers,
            *,
            dense_refiner: object,
            config,
        ) -> None:
            assert received_searchers is searchers
            assert dense_refiner is expected_refiner
            assert config.top_k == 100
            assert config.dense_refine_top_n == 7
            calls["solver_init"] = int(calls["solver_init"]) + 1

        def solve(self, query: str):
            solve_calls = calls["solve"]
            assert isinstance(solve_calls, list)
            solve_calls.append(query)
            if query == duplicate_query:
                return [
                    SimpleNamespace(
                        video_id="L21_V001",
                        frame_idx=101,
                        score=0.91,
                        candidate_id="L21_V001_K000001",
                        timestamp=4.04,
                        evidence={"visual": 0.91},
                    ),
                    SimpleNamespace(
                        video_id="L21_V002",
                        frame_idx=202,
                        score=0.82,
                        candidate_id="L21_V002_K000002",
                        timestamp=8.08,
                        evidence={"asr": 0.82},
                    ),
                ]
            assert query == unique_query
            return [
                SimpleNamespace(
                    video_id="L21_V003",
                    frame_idx=303,
                    score=0.73,
                    candidate_id="L21_V003_K000003",
                    timestamp=12.12,
                    evidence={},
                )
            ]

    monkeypatch.setattr(module, "load_config", fake_load_config)
    monkeypatch.setattr(module, "load_search_fields", fake_load_search_fields)
    monkeypatch.setattr(module, "build_text_searchers", fake_build_text_searchers)
    monkeypatch.setattr(module, "build_dense_refiner", fake_build_dense_refiner)
    monkeypatch.setattr(module, "KISSolver", FakeSolver)

    manifest = module.run_batch(
        input_dir,
        output_dir,
        submission_dir,
        tmp_path / "config.toml",
        dense_refine=True,
        dense_top_n=7,
    )

    assert calls == {
        "load_config": 1,
        "load_search_fields": 1,
        "build_text_searchers": 1,
        "build_dense_refiner": 1,
        "solver_init": 1,
        "solve": [duplicate_query, unique_query],
    }
    assert sorted(path.name for path in output_dir.glob("*.json")) == [
        "manifest.json",
        "query-p1-1-kis.json",
        "query-p1-10-kis.json",
        "query-p1-2-kis.json",
    ]
    assert sorted(path.name for path in submission_dir.glob("*.csv")) == [
        "query-1-kis.csv",
        "query-10-kis.csv",
        "query-2-kis.csv",
    ]
    expected_duplicate_csv = "L21_V001,101\nL21_V002,202\n"
    assert (submission_dir / "query-1-kis.csv").read_text(
        encoding="utf-8"
    ) == expected_duplicate_csv
    assert (submission_dir / "query-10-kis.csv").read_text(
        encoding="utf-8"
    ) == expected_duplicate_csv
    assert (submission_dir / "query-2-kis.csv").read_text(encoding="utf-8") == "L21_V003,303\n"

    first_payload = json.loads((output_dir / "query-p1-1-kis.json").read_text(encoding="utf-8"))
    duplicate_payload = json.loads(
        (output_dir / "query-p1-10-kis.json").read_text(encoding="utf-8")
    )
    assert first_payload["query_id"] == "query-p1-1-kis"
    assert duplicate_payload["query_id"] == "query-p1-10-kis"
    assert first_payload["query"] == duplicate_payload["query"] == duplicate_query
    assert first_payload["predictions"] == duplicate_payload["predictions"]

    persisted_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert persisted_manifest == manifest
    assert not stale_json.exists()
    assert not stale_csv.exists()
    assert set(manifest["removed_stale_files"]) == {
        str(stale_json.resolve()),
        str(stale_csv.resolve()),
    }
    assert manifest["query_count"] == 3
    assert manifest["unique_query_text_count"] == 2
    assert manifest["source_fingerprints"]["query_batch_sha256"]
    assert [item["query_id"] for item in manifest["queries"]] == [
        "query-p1-1-kis",
        "query-p1-2-kis",
        "query-p1-10-kis",
    ]
    assert [Path(item["submission_csv"]).name for item in manifest["queries"]] == [
        "query-1-kis.csv",
        "query-2-kis.csv",
        "query-10-kis.csv",
    ]
    assert [item["reused_identical_query"] for item in manifest["queries"]] == [
        False,
        False,
        True,
    ]


def test_parse_args_uses_full_dataset_batch_config_by_default() -> None:
    module = _load_script()

    args = module.parse_args([])

    assert args.config == "configs/batch1_full.toml"


def test_run_batch_rejects_submission_name_collisions_before_loading_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different source pages must never silently overwrite one official CSV."""

    module = _load_script()
    input_dir = tmp_path / "queries"
    input_dir.mkdir()
    (input_dir / "query-p1-7-kis.txt").write_text("first scene", encoding="utf-8")
    (input_dir / "query-p2-7-kis.txt").write_text("second scene", encoding="utf-8")

    def fail_if_runtime_loads(_path: Path):
        pytest.fail("submission-name collisions must be validated before runtime loading")

    monkeypatch.setattr(module, "load_config", fail_if_runtime_loads)

    with pytest.raises(ValueError, match="same submission filename"):
        module.run_batch(
            input_dir,
            tmp_path / "debug",
            tmp_path / "submission",
            tmp_path / "config.toml",
        )


@pytest.mark.parametrize("limit", [0, 101])
def test_run_batch_rejects_limits_outside_submission_contract(tmp_path: Path, limit: int) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="between 1 and 100"):
        module.run_batch(
            tmp_path / "queries",
            tmp_path / "debug",
            tmp_path / "submission",
            tmp_path / "config.toml",
            limit=limit,
        )
