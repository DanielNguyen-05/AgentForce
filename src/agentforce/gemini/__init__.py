"""Gemini-backed visual question answering.

The package keeps the Google SDK behind a lazy adapter so the rest of the
project, including all tests, works without installing ``google-genai``.
"""

from .cache import JsonFileCache, build_cache_key
from .client import (
    GeminiClientConfig,
    GeminiDependencyError,
    GeminiOutputTruncatedError,
    GeminiQAClient,
    GeminiResponseError,
    GoogleGenAITransport,
)
from .schemas import FrameCandidate, QAVerification, QAVerificationRequest
from .verifier import QAVerifier

__all__ = [
    "FrameCandidate",
    "GeminiClientConfig",
    "GeminiDependencyError",
    "GeminiOutputTruncatedError",
    "GeminiQAClient",
    "GeminiResponseError",
    "GoogleGenAITransport",
    "JsonFileCache",
    "QAVerification",
    "QAVerificationRequest",
    "QAVerifier",
    "build_cache_key",
]
