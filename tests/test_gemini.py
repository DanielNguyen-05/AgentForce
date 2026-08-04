from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from agentforce.config import AppConfig, GeminiConfig, PathConfig
from agentforce.gemini import (
    FrameCandidate,
    GeminiClientConfig,
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
        ),
    )

    verifier = build_gemini_verifier(config)

    assert verifier.prompt_version == "qa-runtime-test"
    assert verifier.client.transport._api_key == "secret-value"
    assert verifier.client.transport._timeout_seconds == 17.5


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
