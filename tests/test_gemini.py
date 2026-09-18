from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentforce.config import AppConfig, GeminiConfig, PathConfig
from agentforce.gemini import (
    FrameCandidate,
    GeminiClientConfig,
    GeminiOutputTruncatedError,
    GeminiQAClient,
    GeminiResponseError,
    GoogleGenAITransport,
    JsonFileCache,
    QAVerification,
    QAVerificationRequest,
    QAVerifier,
)
from agentforce.gemini.prompts import build_qa_prompt
from agentforce.runtime import build_gemini_verifier

VALID_RESPONSE = {
    "answerable": True,
    "answer": "màu xanh",
    "normalized_answer": "xanh",
    "answer_type": "color",
    "supporting_candidate_id": "C02",
    "confidence": 0.91,
    "evidence": ["Chiếc ly trong C02 có màu xanh."],
    "uncertainty_reason": None,
}


class FakeTransport:
    def __init__(self, responses: Sequence[Mapping[str, Any] | str | BaseException]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Mapping[str, Any] | str:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _sdk_config_value(config: Any, name: str) -> Any:
    """Read a GenerateContentConfig represented as either an SDK object or dict."""

    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name)


def _thinking_level(value: Any) -> str | None:
    """Normalize SDK enum/object/dict variants used by different SDK releases."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("thinking_level")
    else:
        value = getattr(value, "thinking_level", value)
    enum_value = getattr(value, "value", value)
    return str(enum_value).rsplit(".", maxsplit=1)[-1].lower()


def _request(tmp_path: Path) -> QAVerificationRequest:
    first = tmp_path / "c01.jpg"
    second = tmp_path / "c02.jpg"
    first.write_bytes(b"fake-jpeg-one")
    second.write_bytes(b"fake-jpeg-two")
    return QAVerificationRequest(
        query_id="Q001",
        question="Chiếc ly có màu gì?",
        retrieval_context="Người phụ nữ mặc váy đỏ đang cầm ly.",
        candidates=(
            FrameCandidate("C01", "L01_V001", 100, 4.0, first, retrieval_score=0.8),
            FrameCandidate(
                "C02",
                "L01_V001",
                120,
                4.8,
                second,
                retrieval_score=0.9,
                ocr_text="LIVE",
            ),
        ),
    )


def test_schema_and_prompt_are_grounded_in_candidate_ids(tmp_path: Path) -> None:
    request = _request(tmp_path)
    prompt = build_qa_prompt(request)
    assert "CANDIDATE C01" in prompt
    assert "CANDIDATE C02" in prompt
    assert "Chiếc ly có màu gì?" in prompt
    assert "caption" not in prompt.casefold()
    assert QAVerification.json_schema()["additionalProperties"] is False
    assert QAVerification.json_schema()["properties"]["evidence"]["maxItems"] == 3


def test_prompt_version_is_configurable_and_part_of_cache_identity(tmp_path: Path) -> None:
    request = _request(tmp_path)
    transport = FakeTransport([VALID_RESPONSE, VALID_RESPONSE])
    client = GeminiQAClient(GeminiClientConfig(max_attempts=1), transport=transport)
    cache = JsonFileCache(tmp_path / "cache")

    QAVerifier(client, cache=cache, prompt_version="qa-test-v1").verify(request)
    QAVerifier(client, cache=cache, prompt_version="qa-test-v2").verify(request)

    assert len(transport.calls) == 2
    assert "Prompt version: qa-test-v1" in transport.calls[0]["prompt"]
    assert "Prompt version: qa-test-v2" in transport.calls[1]["prompt"]


def test_google_transport_converts_timeout_seconds_to_sdk_milliseconds(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeHttpOptions:
        def __init__(self, *, timeout: int) -> None:
            self.timeout = timeout

    class FakeTypes:
        HttpOptions = FakeHttpOptions

    class FakeGenAI:
        @staticmethod
        def Client(**kwargs: Any) -> object:
            captured.update(kwargs)
            return object()

    transport = GoogleGenAITransport(api_key="test-key", timeout_seconds=12.5)
    monkeypatch.setattr(transport, "_imports", lambda: (FakeGenAI, FakeTypes))

    transport._get_client()
    assert captured["api_key"] == "test-key"
    assert captured["http_options"].timeout == 12_500


def test_google_transport_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        GoogleGenAITransport(timeout_seconds=0)


def test_client_defaults_leave_temperature_unset_and_allow_structured_output_room() -> None:
    config = GeminiClientConfig()

    assert config.temperature is None
    assert config.max_output_tokens == 2048
    assert config.max_retry_output_tokens == 8192
    assert config.thinking_level == "minimal"


def test_google_transport_wires_structured_output_and_thinking_config() -> None:
    captured: dict[str, Any] = {}

    class FakePart:
        @staticmethod
        def from_text(*, text: str) -> tuple[str, str]:
            return ("text", text)

    class FakeTypes:
        Part = FakePart

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs: Any) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                parsed=VALID_RESPONSE,
                text=None,
                candidates=[
                    SimpleNamespace(finish_reason="STOP", finish_message="completed")
                ],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=120,
                    candidates_token_count=55,
                    total_token_count=175,
                    cached_content_token_count=None,
                    thoughts_token_count=10,
                ),
            )

    client = SimpleNamespace(models=FakeModels())
    transport = GoogleGenAITransport(client=client)
    transport._get_client = lambda: (client, FakeTypes)  # type: ignore[method-assign]

    result = transport.generate(
        model="gemini-test",
        prompt="question",
        system_instruction="system",
        frames=(),
        response_schema=QAVerification.json_schema(),
        temperature=None,
        max_output_tokens=2048,
        thinking_level="minimal",
    )

    sdk_config = captured["config"]
    assert result == VALID_RESPONSE
    assert _sdk_config_value(sdk_config, "temperature") is None
    assert _sdk_config_value(sdk_config, "max_output_tokens") == 2048
    assert _sdk_config_value(sdk_config, "response_mime_type") == "application/json"
    assert _thinking_level(_sdk_config_value(sdk_config, "thinking_config")) == "minimal"
    assert transport.last_diagnostics["finish_reason"] == "STOP"
    assert transport.last_diagnostics["finish_message"] == "completed"
    assert transport.last_diagnostics["usage"]["thoughts_token_count"] == 10


def test_google_transport_reports_max_tokens_as_explicit_truncation() -> None:
    class FakePart:
        @staticmethod
        def from_text(*, text: str) -> tuple[str, str]:
            return ("text", text)

    class FakeTypes:
        Part = FakePart

    response = SimpleNamespace(
        parsed=None,
        text='{"answerable": true, "answer": "bị cắt',
        candidates=[
            SimpleNamespace(
                finish_reason="MAX_TOKENS",
                finish_message="Output token limit reached",
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=500,
            candidates_token_count=2048,
            total_token_count=2548,
            cached_content_token_count=None,
            thoughts_token_count=1900,
        ),
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_: response)
    )
    transport = GoogleGenAITransport(client=client)
    transport._get_client = lambda: (client, FakeTypes)  # type: ignore[method-assign]

    with pytest.raises(GeminiOutputTruncatedError) as caught:
        transport.generate(
            model="gemini-test",
            prompt="question",
            system_instruction="system",
            frames=(),
            response_schema=QAVerification.json_schema(),
            temperature=None,
            max_output_tokens=2048,
            thinking_level="minimal",
        )

    assert caught.value.retryable is True
    assert caught.value.diagnostics["finish_reason"] == "MAX_TOKENS"
    assert caught.value.diagnostics["finish_message"] == "Output token limit reached"
    assert caught.value.diagnostics["usage"]["candidates_token_count"] == 2048
    assert transport.last_diagnostics == caught.value.diagnostics


def test_runtime_wires_api_key_timeout_and_prompt_version(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEST_GEMINI_API_KEY", "secret-value")
    config = AppConfig(
        paths=PathConfig(
            dataset_root=tmp_path / "dataset",
            artifacts_root=tmp_path / "artifacts",
            outputs_root=tmp_path / "outputs",
        ),
        gemini=GeminiConfig(
            api_key_env="TEST_GEMINI_API_KEY",
            timeout_seconds=17.5,
            prompt_version="qa-runtime-test",
            max_output_tokens=3072,
            max_retry_output_tokens=12288,
            thinking_level="low",
        ),
    )

    verifier = build_gemini_verifier(config)

    assert verifier.prompt_version == "qa-runtime-test"
    assert verifier.client.transport._api_key == "secret-value"
    assert verifier.client.transport._timeout_seconds == 17.5
    assert verifier.client.config.max_output_tokens == 3072
    assert verifier.client.config.max_retry_output_tokens == 12288
    assert verifier.client.config.thinking_level == "low"


def test_runtime_missing_api_key_explains_project_env_setup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TEST_MISSING_GEMINI_KEY", raising=False)
    config = AppConfig(
        paths=PathConfig(
            dataset_root=tmp_path / "dataset",
            artifacts_root=tmp_path / "artifacts",
            outputs_root=tmp_path / "outputs",
        ),
        gemini=GeminiConfig(api_key_env="TEST_MISSING_GEMINI_KEY"),
    )

    with pytest.raises(ValueError, match=r"\.env"):
        build_gemini_verifier(config)


def test_cached_verifier_calls_transport_only_once_and_resolves_local_frame(tmp_path: Path) -> None:
    request = _request(tmp_path)
    transport = FakeTransport([VALID_RESPONSE])
    client = GeminiQAClient(
        GeminiClientConfig(max_attempts=1),
        transport=transport,
        sleep=lambda _: None,
    )
    verifier = QAVerifier(client, cache=JsonFileCache(tmp_path / "cache"))

    first = verifier.verify(request)
    second = verifier.verify(request)

    assert len(transport.calls) == 1
    assert first.supporting_video_id == "L01_V001"
    assert first.supporting_frame_idx == 120
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.supporting_frame_idx == 120


def test_image_content_change_invalidates_cache(tmp_path: Path) -> None:
    request = _request(tmp_path)
    transport = FakeTransport([VALID_RESPONSE, VALID_RESPONSE])
    client = GeminiQAClient(GeminiClientConfig(max_attempts=1), transport=transport)
    verifier = QAVerifier(client, cache=JsonFileCache(tmp_path / "cache"))
    verifier.verify(request)
    Path(request.candidates[0].image_path).write_bytes(b"changed")
    verifier.verify(request)
    assert len(transport.calls) == 2


def test_client_retries_transport_and_invalid_json() -> None:
    transport = FakeTransport([RuntimeError("busy"), "not-json", VALID_RESPONSE])
    sleeps: list[float] = []
    client = GeminiQAClient(
        GeminiClientConfig(
            max_attempts=3,
            initial_backoff_seconds=1,
            max_backoff_seconds=5,
            jitter_ratio=0,
        ),
        transport=transport,
        sleep=sleeps.append,
    )
    response = client.generate(prompt="question", system_instruction="system", frames=())
    assert response == VALID_RESPONSE
    assert sleeps == [1.0, 2.0]


def test_client_doubles_output_budget_after_truncation_and_keeps_attempt_history() -> None:
    truncation = GeminiOutputTruncatedError(
        "Gemini output was truncated",
        diagnostics={
            "finish_reason": "MAX_TOKENS",
            "finish_message": "Output token limit reached",
            "usage": {"candidates_token_count": 2048},
        },
    )
    transport = FakeTransport([truncation, VALID_RESPONSE])
    sleeps: list[float] = []
    client = GeminiQAClient(
        GeminiClientConfig(
            max_attempts=3,
            initial_backoff_seconds=1,
            jitter_ratio=0,
            max_output_tokens=2048,
            max_retry_output_tokens=8192,
        ),
        transport=transport,
        sleep=sleeps.append,
    )

    response = client.generate(prompt="question", system_instruction="system", frames=())

    assert response == VALID_RESPONSE
    assert [call["max_output_tokens"] for call in transport.calls] == [2048, 4096]
    assert all(call["thinking_level"] == "minimal" for call in transport.calls)
    assert sleeps == [1.0]
    assert client.last_diagnostics["attempts"] == 2
    assert client.last_diagnostics["max_output_tokens"] == 4096
    assert client.last_diagnostics["attempt_history"] == [
        {
            "attempt": 1,
            "max_output_tokens": 2048,
            "status": "error",
            "error_type": "GeminiOutputTruncatedError",
            "error": "Gemini output was truncated",
            "transport": truncation.diagnostics,
        },
        {
            "attempt": 2,
            "max_output_tokens": 4096,
            "status": "success",
            "transport": {},
        },
    ]


def test_client_never_increases_output_budget_above_retry_cap() -> None:
    transport = FakeTransport(
        [
            GeminiOutputTruncatedError("cut one"),
            GeminiOutputTruncatedError("cut two"),
            VALID_RESPONSE,
        ]
    )
    client = GeminiQAClient(
        GeminiClientConfig(
            max_attempts=3,
            initial_backoff_seconds=0,
            max_output_tokens=6000,
            max_retry_output_tokens=8192,
        ),
        transport=transport,
        sleep=lambda _: None,
    )

    client.generate(prompt="question", system_instruction="system", frames=())

    assert [call["max_output_tokens"] for call in transport.calls] == [6000, 8192, 8192]


def test_client_fails_fast_for_nonretryable_response_error() -> None:
    error = GeminiResponseError(
        "Gemini blocked the response",
        retryable=False,
        diagnostics={"finish_reason": "SAFETY"},
    )
    transport = FakeTransport([error, VALID_RESPONSE])
    sleeps: list[float] = []
    client = GeminiQAClient(
        GeminiClientConfig(max_attempts=3),
        transport=transport,
        sleep=sleeps.append,
    )

    with pytest.raises(GeminiResponseError, match="blocked"):
        client.generate(prompt="question", system_instruction="system", frames=())

    assert len(transport.calls) == 1
    assert sleeps == []
    assert client.last_diagnostics["attempts"] == 1
    assert client.last_diagnostics["attempt_history"][0]["transport"] == {
        "finish_reason": "SAFETY"
    }


def test_verifier_audits_failed_request_without_populating_cache(tmp_path: Path) -> None:
    error = GeminiResponseError(
        "blocked",
        retryable=False,
        diagnostics={"finish_reason": "SAFETY"},
    )
    cache = JsonFileCache(tmp_path / "cache")
    audit_path = tmp_path / "gemini" / "calls.jsonl"
    verifier = QAVerifier(
        GeminiQAClient(
            GeminiClientConfig(max_attempts=3),
            transport=FakeTransport([error]),
            sleep=lambda _: None,
        ),
        cache=cache,
        audit_path=audit_path,
    )

    with pytest.raises(GeminiResponseError, match="blocked"):
        verifier.verify(_request(tmp_path))

    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["finish_reason"] == "SAFETY"
    assert rows[0]["error_type"] == "GeminiResponseError"
    assert list((tmp_path / "cache").glob("*.json")) == []


def test_unknown_supporting_candidate_is_rejected(tmp_path: Path) -> None:
    response = dict(VALID_RESPONSE, supporting_candidate_id="hallucinated")
    client = GeminiQAClient(
        GeminiClientConfig(max_attempts=1),
        transport=FakeTransport([response]),
    )
    with pytest.raises(GeminiResponseError, match="unknown supporting_candidate_id"):
        QAVerifier(client).verify(_request(tmp_path))


def test_answerable_result_requires_answer_and_support() -> None:
    with pytest.raises(ValueError, match="requires answer"):
        QAVerification.from_mapping(
            dict(
                VALID_RESPONSE,
                answer=None,
                normalized_answer=None,
            )
        )
