from agentforce.utils.io import atomic_write_json, read_jsonl, stable_hash, write_jsonl


def test_json_helpers(tmp_path) -> None:
    json_path = atomic_write_json(tmp_path / "one.json", {"text": "Đà Nẵng"})
    assert "Đà Nẵng" in json_path.read_text(encoding="utf-8")

    rows = [{"id": 1}, {"id": 2}]
    assert write_jsonl(tmp_path / "rows.jsonl", rows) == 2
    assert list(read_jsonl(tmp_path / "rows.jsonl")) == rows
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
