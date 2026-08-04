from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "check_environment.py",
    "validate_dataset.py",
    "validate_artifacts.py",
    "export_timelines.py",
    "transcribe_videos.py",
    "smoke_phowhisper.py",
    "run_ocr.py",
    "normalize_objects.py",
    "build_windows.py",
    "build_visual_indexes.py",
    "build_text_indexes.py",
    "run_search.py",
    "run_kis.py",
    "run_qa.py",
    "run_trake.py",
    "visualize_results.py",
    "verify_qa_request.py",
    "evaluate_results.py",
    "write_submission.py",
)


def test_project_has_only_direct_python_entry_files() -> None:
    assert not (PROJECT_ROOT / "src" / "agentforce" / "cli.py").exists()
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project.scripts]" not in pyproject
    assert "agentforce.cli" not in pyproject
    for name in SCRIPT_NAMES:
        source = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "agentforce.cli" not in source
        assert "[RUN]" in source


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_python_script_help_runs_without_project_install(name: str) -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / name), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.casefold()


def test_run_qa_defaults_to_ranked_competition_mode(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "scripts"))
    namespace = runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_qa.py"))

    default_args = namespace["parse_args"](["--query", "cảnh cần tìm?"])
    smoke_args = namespace["parse_args"](["--query", "cảnh cần tìm?", "--single-answer"])

    assert default_args.single_answer is False
    assert default_args.answer_limit == 100
    assert smoke_args.single_answer is True


def test_run_qa_preflights_gemini_key_before_loading_search_models(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "scripts"))
    namespace = runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_qa.py"))
    solve_qa = namespace["solve_qa"]
    call_order: list[str] = []
    config = SimpleNamespace(gemini=SimpleNamespace(max_candidates=12))

    def missing_key(_config) -> None:
        call_order.append("gemini_preflight")
        raise ValueError("Missing GEMINI_API_KEY; fill project-root .env")

    def load_search_models(_config) -> None:
        call_order.append("search_models")
        raise AssertionError("Search models must not load when the Gemini key is missing")

    monkeypatch.setitem(solve_qa.__globals__, "load_config", lambda _path: config)
    monkeypatch.setitem(solve_qa.__globals__, "build_gemini_verifier", missing_key)
    monkeypatch.setitem(solve_qa.__globals__, "load_search_fields", load_search_models)

    with pytest.raises(ValueError, match=r"\.env"):
        solve_qa(
            "một cảnh cần tìm",
            question="Vật thể có màu gì?",
            query_id="QA001",
            max_candidates=None,
            batch_size=4,
            answer_limit=100,
            config_path=tmp_path / "unused.toml",
        )

    assert call_order == ["gemini_preflight"]
