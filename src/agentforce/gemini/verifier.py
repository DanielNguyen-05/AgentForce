"""High-level cached Q&A verification over locally retrieved frames."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cache import JsonFileCache, build_cache_key
from .client import GeminiQAClient, GeminiResponseError
from .prompts import PROMPT_VERSION, SYSTEM_INSTRUCTION, build_qa_prompt
from .schemas import QAVerification, QAVerificationRequest


class QAVerifier:
    def __init__(
        self,
        client: GeminiQAClient,
        *,
        cache: JsonFileCache | None = None,
        audit_path: str | Path | None = None,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self.client = client
        self.cache = cache
        self.audit_path = Path(audit_path) if audit_path is not None else None
        self.prompt_version = prompt_version.strip()
        if not self.prompt_version:
            raise ValueError("prompt_version must not be empty")
        self.last_diagnostics: dict[str, Any] = {}

    def _audit(
        self,
        request: QAVerificationRequest,
        *,
        cache_hit: bool,
        status: str = "success",
        error: BaseException | None = None,
    ) -> None:
        self.last_diagnostics = {
            **dict(self.client.last_diagnostics),
            "cache_hit": cache_hit,
            "status": status,
        }
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "query_id": request.query_id,
            "model": self.client.config.model,
            "candidate_count": len(request.candidates),
            **self.last_diagnostics,
        }
        if error is not None:
            row["error_type"] = type(error).__name__
            row["error"] = str(error)[:1000]
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    @staticmethod
    def _image_digests(request: QAVerificationRequest) -> dict[str, str]:
        digests: dict[str, str] = {}
        for candidate in request.candidates:
            path = Path(candidate.image_path)
            try:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
            except OSError as exc:
                raise ValueError(f"Cannot read candidate image {path}: {exc}") from exc
            digests[candidate.candidate_id] = digest
        return digests

    def _cache_key(self, request: QAVerificationRequest, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.client.config.model,
            "generation": {
                "temperature": self.client.config.temperature,
                "max_output_tokens": self.client.config.max_output_tokens,
                "max_retry_output_tokens": self.client.config.max_retry_output_tokens,
                "thinking_level": self.client.config.thinking_level,
            },
            "prompt_version": self.prompt_version,
            "system_instruction": SYSTEM_INSTRUCTION,
            "prompt": prompt,
            "query_id": request.query_id,
            "language": request.language,
            "candidates": [candidate.metadata_dict() for candidate in request.candidates],
        }
        return build_cache_key(payload, self._image_digests(request))

    @staticmethod
    def _resolve_support(
        result: QAVerification,
        request: QAVerificationRequest,
        *,
        cache_hit: bool,
    ) -> QAVerification:
        if result.supporting_candidate_id is None:
            # Unanswerable results intentionally carry no frame support.
            return QAVerification(
                answerable=result.answerable,
                answer=result.answer,
                normalized_answer=result.normalized_answer,
                answer_type=result.answer_type,
                supporting_candidate_id=None,
                confidence=result.confidence,
                evidence=result.evidence,
                uncertainty_reason=result.uncertainty_reason,
                cache_hit=cache_hit,
            )
        candidates = {candidate.candidate_id: candidate for candidate in request.candidates}
        candidate = candidates.get(result.supporting_candidate_id)
        if candidate is None:
            raise GeminiResponseError(
                "Gemini selected an unknown supporting_candidate_id: "
                f"{result.supporting_candidate_id}"
            )
        return result.with_local_support(candidate, cache_hit=cache_hit)

    def verify(self, request: QAVerificationRequest) -> QAVerification:
        prompt = build_qa_prompt(request, prompt_version=self.prompt_version)
        cache_key: str | None = None
        if self.cache is not None:
            cache_key = self._cache_key(request, prompt)
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached_response = cached.get("response")
                if not isinstance(cached_response, dict):
                    raise ValueError(f"Gemini cache response is invalid for key {cache_key}")
                result = QAVerification.from_mapping(cached_response, cache_hit=True)
                diagnostics = cached.get("diagnostics")
                self.client.last_diagnostics = dict(diagnostics) if isinstance(diagnostics, dict) else {}
                self._audit(request, cache_hit=True)
                return self._resolve_support(result, request, cache_hit=True)

        try:
            response = self.client.generate(
                prompt=prompt,
                system_instruction=SYSTEM_INSTRUCTION,
                frames=request.candidates,
            )
            result = QAVerification.from_mapping(response)
            resolved = self._resolve_support(result, request, cache_hit=False)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            self._audit(request, cache_hit=False, status="error", error=exc)
            raise
        self._audit(request, cache_hit=False)
        if self.cache is not None and cache_key is not None:
            self.cache.set(
                cache_key,
                {
                    "model": self.client.config.model,
                    "prompt_version": self.prompt_version,
                    "query_id": request.query_id,
                    "response": response,
                    "diagnostics": self.client.last_diagnostics,
                },
            )
        return resolved
