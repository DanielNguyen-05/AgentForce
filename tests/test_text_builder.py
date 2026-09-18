from agentforce.embeddings.encoders import HashingTextEncoder
from agentforce.indexing.build_text import build_text_index
from agentforce.indexing.numpy_index import NumpyIndex


def test_build_text_index_skips_empty_rows(tmp_path) -> None:
    records = [
        {"segment_id": "a", "text": "xe đạp trên đường"},
        {"segment_id": "b", "text": ""},
        {"segment_id": "c", "text": "người mở laptop"},
    ]
    manifest = build_text_index(
        records,
        tmp_path,
        "asr",
        HashingTextEncoder(dimension=32),
        id_field="segment_id",
        dtype="float32",
    )
    assert manifest["row_count"] == 2
    index = NumpyIndex(tmp_path / "asr.npy", tmp_path / "asr.jsonl")
    assert index.search(HashingTextEncoder(32).encode(["laptop"])[0], 1)[0].vector_id == "c"
