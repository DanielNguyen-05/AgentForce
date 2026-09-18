"""OpenCLIP semantic scorer for decoded frames."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from agentforce.errors import OptionalDependencyError
from .refinement import DecodedFrame


class OpenCLIPFrameScorer:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        *,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError as exc:
            raise OptionalDependencyError(
                "CLIP frame scoring requires `pip install -e '.[vision]'`"
            ) from exc
        self._torch = torch
        self._image_type = Image
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._device = device
        self._batch_size = batch_size

    def _image(self, frame: DecodedFrame) -> Any:
        array = np.asarray(frame.payload)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ValueError("Decoded frame payload must be HxWx3/4 image data")
        if frame.metadata.get("color_space") == "bgr":
            array = array[..., :3][..., ::-1]
        return self._image_type.fromarray(array.astype(np.uint8, copy=False))

    def score(self, query: str, frames: Sequence[DecodedFrame]) -> Sequence[float]:
        if not frames:
            return []
        tokens = self._tokenizer([query]).to(self._device)
        with self._torch.inference_mode():
            text = self._model.encode_text(tokens, normalize=True)
            scores: list[float] = []
            for start in range(0, len(frames), self._batch_size):
                batch = self._torch.stack(
                    [self._preprocess(self._image(frame)) for frame in frames[start : start + self._batch_size]]
                ).to(self._device)
                images = self._model.encode_image(batch, normalize=True)
                scores.extend((images @ text.T).squeeze(1).detach().cpu().tolist())
        return [float(value) for value in scores]

