"""Selective keyframe captioning with a mockable optional Gemini backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any, Mapping, Protocol

from agentforce.gemini.cache import JsonFileCache, build_cache_key


CAPTION_PROMPT_VERSION = "keyframe-caption-v1"
CAPTION_PROMPT = """Analyze only this video keyframe. Return structured JSON.
Describe visible people, actions, scene, objects, attributes, and readable text.
Use concise Vietnamese. Do not infer events outside the image.
"""


@dataclass(frozen=True, slots=True)
class CaptionRecord:
    keyframe_uid: str
    caption_text: str
    people: tuple[str, ...]
    actions: tuple[str, ...]
    scene: str
    objects: tuple[str, ...]
    attributes: tuple[str, ...]
    visible_text: tuple[str, ...]
    model: str
    prompt_version: str = CAPTION_PROMPT_VERSION
    source_path: str | None = None
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CaptionBackend(Protocol):
    model: str

    def caption(self, image_path: Path) -> Mapping[str, Any]: ...


def caption_schema() -> dict[str, Any]:
    array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "caption_text": {"type": "string"},
            "people": array,
            "actions": array,
            "scene": {"type": "string"},
            "objects": array,
            "attributes": array,
            "visible_text": array,
        },
        "required": [
            "caption_text",
            "people",
            "actions",
            "scene",
            "objects",
            "attributes",
            "visible_text",
        ],
    }


class GeminiCaptionBackend:
    def __init__(self, *, model: str = "gemini-2.5-flash", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key
        self._client: Any | None = None

    def _sdk(self) -> tuple[Any, Any]:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Gemini captioning requires `pip install -e '.[gemini]'`"
            ) from exc
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client, types

    def caption(self, image_path: Path) -> Mapping[str, Any]:
        client, types = self._sdk()
        mime_type, _ = mimetypes.guess_type(image_path.name)
        if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError(f"Unsupported caption image: {image_path}")
        response = client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_text(text=CAPTION_PROMPT),
                types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type),
            ],
            config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
                "response_json_schema": caption_schema(),
            },
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, Mapping):
            return parsed
        text = getattr(response, "text", "")
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("Gemini caption response must be an object")
        return value


class CaptionProcessor:
    def __init__(self, backend: CaptionBackend, *, cache: JsonFileCache | None = None) -> None:
        self.backend = backend
        self.cache = cache

    @staticmethod
    def _validate(value: Mapping[str, Any]) -> dict[str, Any]:
        required_arrays = ("people", "actions", "objects", "attributes", "visible_text")
        caption = str(value.get("caption_text", "")).strip()
        if not caption:
            raise ValueError("Caption response has empty caption_text")
        result: dict[str, Any] = {"caption_text": caption, "scene": str(value.get("scene", "")).strip()}
        for field in required_arrays:
            raw = value.get(field, [])
            if isinstance(raw, str) or not isinstance(raw, list):
                raise ValueError(f"Caption response field {field} must be an array")
            result[field] = tuple(str(item).strip() for item in raw if str(item).strip())
        return result

    def process(self, image_path: str | Path, keyframe_uid: str) -> CaptionRecord:
        source = Path(image_path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        key = build_cache_key(
            {
                "model": self.backend.model,
                "prompt_version": CAPTION_PROMPT_VERSION,
                "keyframe_uid": keyframe_uid,
            },
            {keyframe_uid: digest},
        )
        cached = self.cache.get(key) if self.cache else None
        raw = cached["response"] if cached is not None else self.backend.caption(source)
        parsed = self._validate(raw)
        if self.cache and cached is None:
            self.cache.set(key, {"response": dict(raw)})
        return CaptionRecord(
            keyframe_uid=keyframe_uid,
            caption_text=parsed["caption_text"],
            people=parsed["people"],
            actions=parsed["actions"],
            scene=parsed["scene"],
            objects=parsed["objects"],
            attributes=parsed["attributes"],
            visible_text=parsed["visible_text"],
            model=self.backend.model,
            source_path=str(source),
            cache_hit=cached is not None,
        )

