"""Batch builder for ASR/OCR/caption/metadata text indexes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import os
from datetime import datetime, timezone

import numpy as np

from agentforce.embeddings.encoders import TextEncoder


def build_text_index(
    records: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    name: str,
    encoder: TextEncoder,
    *,
    text_field: str = "text",
    id_field: str = "vector_id",
    batch_size: int = 128,
    dtype: str = "float16",
) -> dict[str, Any]:
    """Encode structured records while preserving row-aligned metadata.

    Empty text rows are skipped. The iterable is consumed once into lightweight
    dictionaries so dimensions and a deterministic output shape are known.
    """

    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        text = str(row.get(text_field, "")).strip()
        if not text:
            continue
        if id_field not in row:
            raise KeyError(f"Record is missing required id field: {id_field}")
        row[text_field] = text
        rows.append(row)
    if not rows:
        raise ValueError("Cannot build a text index with no non-empty text records")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vector_path = output / f"{name}.npy"
    metadata_path = output / f"{name}.jsonl"
    manifest_path = output / f"{name}.manifest.json"
    temporary_vector = output / f".{name}.npy.tmp"
    temporary_metadata = output / f".{name}.jsonl.tmp"

    matrix = np.lib.format.open_memmap(
        temporary_vector,
        mode="w+",
        dtype=np.dtype(dtype),
        shape=(len(rows), int(encoder.dimension)),
    )
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        encoded = encoder.encode([row[text_field] for row in rows[start:stop]])
        if encoded.shape != (stop - start, encoder.dimension):
            raise ValueError(
                f"Encoder returned {encoded.shape}; expected {(stop - start, encoder.dimension)}"
            )
        matrix[start:stop] = encoded.astype(dtype, copy=False)
    matrix.flush()
    del matrix

    with temporary_metadata.open("w", encoding="utf-8") as handle:
        for row in rows:
            stored = dict(row)
            stored["vector_id"] = str(row[id_field])
            handle.write(json.dumps(stored, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary_vector, vector_path)
    os.replace(temporary_metadata, metadata_path)
    manifest = {
        "name": name,
        "vectors_path": vector_path.name,
        "metadata_path": metadata_path.name,
        "row_count": len(rows),
        "dimension": int(encoder.dimension),
        "dtype": np.dtype(dtype).name,
        "normalized": True,
        "text_field": text_field,
        "id_field": id_field,
        "encoder": type(encoder).__name__,
        "encoder_model": getattr(encoder, "model_name", None),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "builder_version": 1,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
