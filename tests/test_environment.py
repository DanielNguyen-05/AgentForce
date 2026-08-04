from __future__ import annotations

import os

from agentforce.environment import load_project_env


def test_load_project_env_reads_explicit_project_root(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("AGENTFORCE_TEST_KEY", raising=False)
    (tmp_path / ".env").write_text("AGENTFORCE_TEST_KEY=from-file\n", encoding="utf-8")

    assert load_project_env(tmp_path) is True
    assert os.environ["AGENTFORCE_TEST_KEY"] == "from-file"


def test_load_project_env_does_not_override_exported_value(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGENTFORCE_TEST_KEY", "from-shell")
    (tmp_path / ".env").write_text("AGENTFORCE_TEST_KEY=from-file\n", encoding="utf-8")

    load_project_env(tmp_path)
    assert os.environ["AGENTFORCE_TEST_KEY"] == "from-shell"


def test_load_project_env_allows_missing_file(tmp_path) -> None:
    assert load_project_env(tmp_path) is False
