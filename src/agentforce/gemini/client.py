"""Retrying structured-output client with an optional Google Gen AI adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
import mimetypes
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from .schemas import FrameCandidate, QAVerification


class GeminiDependencyError(RuntimeError):
    """Raised when the optional SDK is needed but not installed/configured."""


class GeminiResponseError(RuntimeError):
    """Raised when Gemini returns an unusable structured response."""


@dataclass(frozen=True, slots=True)
class GeminiClientConfig:
    model: str = "gemini-2.5-flash"
    max_attempts: int = 3
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 8.0
    jitter_ratio: float = 0.1
    temperature: float = 0.0
    max_output_tokens: int = 512

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Gemini model must not be empty")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.initial_backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("Backoff values must be non-negative")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")


class GeminiTransport(Protocol):
    """Narrow boundary that makes API behavior mockable without the SDK."""

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str,
        frames: Sequence[FrameCandidate],
        response_schema: Mapping[str, Any],
        temperature: float,
        max_output_tokens: int,
    ) -> Mapping[str, Any] | str:
        ...


class GoogleGenAITransport:
    """Adapter for the official ``google-genai`` Python package.

    Imports are intentionally local. Creating or importing AgentForce does not
    require an API key and does not require ``google-genai`` to be installed.
    """

    def __init__(self, *, api_key: str | None = None, client: object | None = None) -> None:
        self._api_key = api_key
        self._client = client
        self.last_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _imports() -> tuple[Any, Any]:
        try:
            from google import genai  # type: ignore[import-not-found]
            from google.genai import types  # type: ignore[import-not-found]
        except ImportError as exc:
            raise GeminiDependencyError(
                "Gemini Q&A requires the optional 'google-genai' package. "
                "Install the project Gemini extra before making API calls."
            ) from exc
        return genai, types

    def _get_client(self) -> tuple[object, Any]:
        genai, types = self._imports()
        if self._client is None:
            # With api_key=None the SDK reads GEMINI_API_KEY/GOOGLE_API_KEY.
            self._client = genai.Client(api_key=self._api_key)
        return self._client, types

    @staticmethod
    def _mime_type(path: Path) -> str:
        guessed, _ = mimetypes.guess_type(path.name)
        if guessed in {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}:
            return guessed
        raise ValueError(f"Unsupported candidate image type: {path}")

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str,
        frames: Sequence[FrameCandidate],
        response_schema: Mapping[str, Any],
        temperature: float,
        max_output_tokens: int,
    ) -> Mapping[str, Any] | str:
        client, types = self._get_client()
        contents: list[Any] = [types.Part.from_text(text=prompt)]
        for frame in frames:
            path = Path(frame.image_path)
            try:
                image_bytes = path.read_bytes()
            except OSError as exc:
                raise ValueError(f"Cannot read candidate image {path}: {exc}") from exc
            contents.append(
                types.Part.from_text(
                    text=(
                        f"FRAME {frame.candidate_id} | video={frame.video_id} | "
                        f"frame_idx={frame.frame_idx} | timestamp={frame.timestamp:.3f}s"
                    )
                )
            )
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=self._mime_type(path)))

        started = time.perf_counter()
        response = client.models.generate_content(  # type: ignore[attr-defined]
            model=model,
            contents=contents,
            config={
                "system_instruction": system_instruction,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json",
                "response_json_schema": dict(response_schema),
            },
        )
        usage = getattr(response, "usage_metadata", None)
        usage_fields = (
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
            "cached_content_token_count",
            "thoughts_token_count",
        )
        self.last_diagnostics = {
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "usage": {
                field: getattr(usage, field, None)
                for field in usage_fields
                if usage is not None and getattr(usage, field, None) is not None
            },
        }
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, Mapping):
            return parsed
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise GeminiResponseError("Gemini returned neither parsed JSON nor response text")
        return text


class GeminiQAClient:
    """Validate responses and retry transient/invalid model calls."""

    def __init__(
        self,
        config: GeminiClientConfig | None = None,
        *,
        transport: GeminiTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self.config = config or GeminiClientConfig()
        self.transport = transport or GoogleGenAITransport()
        self._sleep = sleep
        self._random = random_value
        self.last_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _parse_response(value: Mapping[str, Any] | str) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        if not isinstance(value, str):
            raise GeminiResponseError("Gemini transport returned an unsupported response type")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise GeminiResponseError(f"Gemini response is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise GeminiResponseError("Gemini response JSON must be an object")
        return parsed

    def _delay(self, failed_attempt: int) -> float:
        base = min(
            self.config.max_backoff_seconds,
            self.config.initial_backoff_seconds * (2 ** (failed_attempt - 1)),
        )
        if base == 0:
            return 0.0
        jitter = base * self.config.jitter_ratio * ((2 * self._random()) - 1)
        return max(0.0, base + jitter)

    def generate(
        self,
        *,
        prompt: str,
        system_instruction: str,
        frames: Sequence[FrameCandidate],
    ) -> dict[str, Any]:
        last_error: BaseException | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self.transport.generate(
                    model=self.config.model,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    frames=frames,
                    response_schema=QAVerification.json_schema(),
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                )
                parsed = self._parse_response(response)
                # Validate here so a schema-compliant but semantically invalid
                # model answer is retried before reaching the orchestrator.
                QAVerification.from_mapping(parsed)
                self.last_diagnostics = {
                    **dict(getattr(self.transport, "last_diagnostics", {}) or {}),
                    "attempts": attempt,
                }
                return parsed
            except GeminiDependencyError:
                raise
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                last_error = exc
                if attempt < self.config.max_attempts:
                    self._sleep(self._delay(attempt))
        raise GeminiResponseError(
            f"Gemini request failed after {self.config.max_attempts} attempt(s): {last_error}"
        ) from last_error
