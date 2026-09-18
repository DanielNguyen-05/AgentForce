"""Retrying structured-output client with an optional Google Gen AI adapter."""

from __future__ import annotations

import json
import mimetypes
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .schemas import FrameCandidate, QAVerification


class GeminiDependencyError(RuntimeError):
    """Raised when the optional SDK is needed but not installed/configured."""


class GeminiResponseError(RuntimeError):
    """Raised when Gemini returns an unusable structured response."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})
        self.retryable = retryable


class GeminiOutputTruncatedError(GeminiResponseError):
    """Raised when Gemini stops at the configured output-token limit."""


@dataclass(frozen=True, slots=True)
class GeminiClientConfig:
    model: str = "gemini-3.6-flash"
    max_attempts: int = 3
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 8.0
    jitter_ratio: float = 0.1
    # Gemini 3.6 no longer accepts sampling parameters. Keep this optional for
    # compatibility with older models, but omit it from the default request.
    temperature: float | None = None
    max_output_tokens: int = 2048
    max_retry_output_tokens: int = 8192
    # VQA is short factual classification, so minimal thinking is enough and
    # leaves the output budget available for the structured answer.
    thinking_level: str | None = "minimal"

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
        if self.max_retry_output_tokens < self.max_output_tokens:
            raise ValueError("max_retry_output_tokens must be >= max_output_tokens")
        if self.thinking_level not in {None, "minimal", "low", "medium", "high"}:
            raise ValueError("thinking_level must be minimal, low, medium, high, or None")


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
        temperature: float | None,
        max_output_tokens: int,
        thinking_level: str | None,
    ) -> Mapping[str, Any] | str:
        ...


class GoogleGenAITransport:
    """Adapter for the official ``google-genai`` Python package.

    Imports are intentionally local. Creating or importing AgentForce does not
    require an API key and does not require ``google-genai`` to be installed.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        client: object | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._api_key = api_key
        self._timeout_seconds = float(timeout_seconds)
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
            # The Google Gen AI SDK expects HttpOptions.timeout in milliseconds.
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(timeout=round(self._timeout_seconds * 1000)),
            )
        return self._client, types

    @staticmethod
    def _mime_type(path: Path) -> str:
        guessed, _ = mimetypes.guess_type(path.name)
        if guessed in {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}:
            return guessed
        raise ValueError(f"Unsupported candidate image type: {path}")

    @staticmethod
    def _enum_value(value: object) -> str | None:
        if value is None:
            return None
        raw = getattr(value, "value", value)
        text = str(raw)
        # Be tolerant of older/fake SDK objects whose str() is
        # ``FinishReason.MAX_TOKENS`` rather than simply ``MAX_TOKENS``.
        return text.rsplit(".", 1)[-1]

    @classmethod
    def _response_diagnostics(
        cls,
        response: object,
        *,
        latency_ms: float,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        usage = getattr(response, "usage_metadata", None)
        usage_fields = (
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
            "cached_content_token_count",
            "thoughts_token_count",
        )
        candidate_rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(getattr(response, "candidates", None) or []):
            row = {
                "index": getattr(candidate, "index", index),
                "finish_reason": cls._enum_value(getattr(candidate, "finish_reason", None)),
                "finish_message": getattr(candidate, "finish_message", None),
                "token_count": getattr(candidate, "token_count", None),
            }
            candidate_rows.append({key: value for key, value in row.items() if value is not None})

        prompt_feedback = getattr(response, "prompt_feedback", None)
        feedback: dict[str, Any] = {}
        if prompt_feedback is not None:
            block_reason = cls._enum_value(getattr(prompt_feedback, "block_reason", None))
            block_message = getattr(prompt_feedback, "block_reason_message", None)
            if block_reason is not None:
                feedback["block_reason"] = block_reason
            if block_message is not None:
                feedback["block_reason_message"] = str(block_message)

        diagnostics: dict[str, Any] = {
            "latency_ms": round(latency_ms, 3),
            "requested_max_output_tokens": max_output_tokens,
            "usage": {
                field: getattr(usage, field, None)
                for field in usage_fields
                if usage is not None and getattr(usage, field, None) is not None
            },
            "candidates": candidate_rows,
        }
        if candidate_rows:
            # Mirror the primary candidate at the top level for quick audit
            # inspection while retaining the full list for future multi-candidate
            # compatibility.
            if "finish_reason" in candidate_rows[0]:
                diagnostics["finish_reason"] = candidate_rows[0]["finish_reason"]
            if "finish_message" in candidate_rows[0]:
                diagnostics["finish_message"] = candidate_rows[0]["finish_message"]
        for field in ("response_id", "model_version"):
            value = getattr(response, field, None)
            if value is not None:
                diagnostics[field] = str(value)
        if feedback:
            diagnostics["prompt_feedback"] = feedback
        return diagnostics

    @staticmethod
    def _parsed_mapping(value: object) -> dict[str, Any] | None:
        if isinstance(value, Mapping):
            return dict(value)
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(mode="json")
            if isinstance(dumped, Mapping):
                return dict(dumped)
        return None

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str,
        frames: Sequence[FrameCandidate],
        response_schema: Mapping[str, Any],
        temperature: float | None,
        max_output_tokens: int,
        thinking_level: str | None,
    ) -> Mapping[str, Any] | str:
        # Never attribute diagnostics from a previous call to a request that
        # fails before the SDK returns a response.
        self.last_diagnostics = {}
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

        generation_config: dict[str, Any] = {
            "system_instruction": system_instruction,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "application/json",
            "response_json_schema": dict(response_schema),
        }
        if temperature is not None:
            generation_config["temperature"] = temperature
        if thinking_level is not None:
            generation_config["thinking_config"] = {"thinking_level": thinking_level}

        started = time.perf_counter()
        response = client.models.generate_content(  # type: ignore[attr-defined]
            model=model,
            contents=contents,
            config=generation_config,
        )
        self.last_diagnostics = self._response_diagnostics(
            response,
            latency_ms=(time.perf_counter() - started) * 1000,
            max_output_tokens=max_output_tokens,
        )

        parsed = self._parsed_mapping(getattr(response, "parsed", None))
        if parsed is not None:
            return parsed

        try:
            text = getattr(response, "text", None)
        except Exception as exc:  # SDK property can fail for blocked/no-content responses.
            text = None
            self.last_diagnostics["response_text_error"] = (
                f"{type(exc).__name__}: {str(exc)[:500]}"
            )
        if isinstance(text, str):
            self.last_diagnostics["response_text_chars"] = len(text)

        candidate_rows = self.last_diagnostics.get("candidates", [])
        first_finish_reason = (
            candidate_rows[0].get("finish_reason") if candidate_rows else None
        )
        if first_finish_reason == "MAX_TOKENS":
            # A response can theoretically finish exactly at the boundary and
            # still be valid. Accept that rare case; never try to patch partial
            # JSON by inventing closing quotes/braces.
            if isinstance(text, str) and text.strip():
                try:
                    complete = json.loads(text)
                except json.JSONDecodeError:
                    complete = None
                if isinstance(complete, Mapping):
                    return dict(complete)
            raise GeminiOutputTruncatedError(
                "Gemini stopped at MAX_TOKENS before completing structured JSON "
                f"(max_output_tokens={max_output_tokens})",
                diagnostics=self.last_diagnostics,
            )

        terminal_reasons = {
            "SAFETY",
            "RECITATION",
            "LANGUAGE",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "MALFORMED_FUNCTION_CALL",
            "IMAGE_SAFETY",
            "UNEXPECTED_TOOL_CALL",
            "IMAGE_PROHIBITED_CONTENT",
            "NO_IMAGE",
            "IMAGE_RECITATION",
        }
        if first_finish_reason in terminal_reasons:
            message = candidate_rows[0].get("finish_message") if candidate_rows else None
            detail = f": {message}" if message else ""
            raise GeminiResponseError(
                f"Gemini stopped with finish_reason={first_finish_reason}{detail}",
                diagnostics=self.last_diagnostics,
                retryable=False,
            )
        if not isinstance(text, str) or not text.strip():
            raise GeminiResponseError(
                "Gemini returned neither parsed JSON nor response text"
                + (f" (finish_reason={first_finish_reason})" if first_finish_reason else ""),
                diagnostics=self.last_diagnostics,
            )
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
        attempts_made = 0
        max_output_tokens = self.config.max_output_tokens
        attempt_history: list[dict[str, Any]] = []
        for attempt in range(1, self.config.max_attempts + 1):
            attempts_made = attempt
            try:
                response = self.transport.generate(
                    model=self.config.model,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    frames=frames,
                    response_schema=QAVerification.json_schema(),
                    temperature=self.config.temperature,
                    max_output_tokens=max_output_tokens,
                    thinking_level=self.config.thinking_level,
                )
                parsed = self._parse_response(response)
                # Validate here so a schema-compliant but semantically invalid
                # model answer is retried before reaching the orchestrator.
                try:
                    QAVerification.from_mapping(parsed)
                except (TypeError, ValueError) as exc:
                    raise GeminiResponseError(
                        f"Gemini response failed semantic validation: {exc}"
                    ) from exc
                transport_diagnostics = dict(
                    getattr(self.transport, "last_diagnostics", {}) or {}
                )
                attempt_history.append(
                    {
                        "attempt": attempt,
                        "max_output_tokens": max_output_tokens,
                        "status": "success",
                        "transport": transport_diagnostics,
                    }
                )
                self.last_diagnostics = {
                    **transport_diagnostics,
                    "attempts": attempt,
                    "max_output_tokens": max_output_tokens,
                    "attempt_history": attempt_history,
                }
                return parsed
            except GeminiDependencyError:
                raise
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                last_error = exc
                transport_diagnostics = dict(
                    getattr(exc, "diagnostics", None)
                    or getattr(self.transport, "last_diagnostics", {})
                    or {}
                )
                attempt_history.append(
                    {
                        "attempt": attempt,
                        "max_output_tokens": max_output_tokens,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                        "transport": transport_diagnostics,
                    }
                )
                self.last_diagnostics = {
                    **transport_diagnostics,
                    "status": "error",
                    "attempts": attempt,
                    "max_output_tokens": max_output_tokens,
                    "attempt_history": attempt_history,
                }

                retryable = getattr(exc, "retryable", True)
                if isinstance(exc, (ValueError, TypeError, OSError)) and not isinstance(
                    exc, GeminiResponseError
                ):
                    retryable = False
                if not retryable or attempt >= self.config.max_attempts:
                    break

                if isinstance(exc, GeminiOutputTruncatedError):
                    max_output_tokens = min(
                        self.config.max_retry_output_tokens,
                        max_output_tokens * 2,
                    )
                self._sleep(self._delay(attempt))

        assert last_error is not None
        diagnostics = dict(self.last_diagnostics)
        final_error = GeminiResponseError(
            f"Gemini request failed after {attempts_made} attempt(s): {last_error}",
            diagnostics=diagnostics,
            retryable=bool(getattr(last_error, "retryable", True)),
        )
        raise final_error from last_error
