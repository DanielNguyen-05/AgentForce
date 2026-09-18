"""Replaceable text encoders with lazy optional imports.

``HashingTextEncoder`` exists for tests and pipeline plumbing only. It is not a
semantic model and must not be used to report retrieval quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence
import hashlib
import re

import numpy as np

from agentforce.errors import OptionalDependencyError


class TextEncoder(Protocol):
    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a normalized ``float32`` matrix with one row per text."""


def l2_normalize(matrix: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, epsilon)


@dataclass(slots=True)
class HashingTextEncoder:
    """Deterministic dependency-free encoder for smoke tests."""

    dimension: int = 512

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
                index = int.from_bytes(digest[:8], "little") % self.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                vectors[row, index] += sign
        return l2_normalize(vectors)


class SentenceTransformerEncoder:
    """Multilingual text encoder for ASR/OCR/object/metadata fields."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise OptionalDependencyError(
                "SentenceTransformerEncoder requires `pip install -e '.[semantic]'`"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(
            model_name,
            device=device,
            local_files_only=local_files_only,
        )
        dimension_getter = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self.dimension = int(dimension_getter())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(matrix, dtype=np.float32)


class OpenCLIPTextEncoder:
    """Text tower compatible with the supplied CLIP ViT-B/32 image features."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        *,
        device: str = "cpu",
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise OptionalDependencyError(
                "OpenCLIPTextEncoder requires `pip install -e '.[vision]'`"
            ) from exc
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._torch = torch
        self._model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)
        projection = getattr(self._model, "text_projection", None)
        self.dimension = int(projection.shape[-1]) if projection is not None else 512

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        tokens = self._tokenizer(list(texts)).to(self.device)
        with self._torch.inference_mode():
            features = self._model.encode_text(tokens, normalize=True)
        return features.detach().cpu().numpy().astype(np.float32, copy=False)
