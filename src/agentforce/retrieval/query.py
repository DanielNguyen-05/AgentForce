"""Heuristic query parsing and replaceable query-expansion interfaces."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from typing import Protocol

from .types import ParsedQuery, QueryEvent, QueryVariant, TaskType

_SEQUENCE_PATTERN = re.compile(
    r"\s*(?:\s+[-=]>\s+|\b(?:sau đó|sau do|tiếp theo|tiep theo|rồi|roi|"
    r"kế tiếp|ke tiep|finally|then|next)\b)\s*",
    flags=re.IGNORECASE,
)
_NUMBERED_EVENT_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+(.+?)(?=(?:\n\s*(?:\d+[.)]|[-*])\s+)|$)",
    flags=re.DOTALL,
)
_VIETNAMESE_MARKERS = {
    "cảnh",
    "người",
    "đang",
    "sau",
    "trước",
    "màu",
    "bao nhiêu",
    "ở đâu",
    "là gì",
}


class QueryParser(Protocol):
    def parse(
        self, text: str, *, task_type: TaskType | str | None = None
    ) -> ParsedQuery: ...


class QueryExpander(Protocol):
    def expand(self, query: ParsedQuery) -> Sequence[QueryVariant]: ...


def _language(text: str) -> str:
    lowered = text.casefold()
    if any(marker in lowered for marker in _VIETNAMESE_MARKERS):
        return "vi"
    decomposed = unicodedata.normalize("NFD", lowered)
    if any(char in decomposed for char in "̛̆̂") or "đ" in lowered:
        return "vi"
    return "en" if re.search(r"[a-z]", lowered) else "unknown"


def _expected_answer_type(text: str) -> str | None:
    lowered = text.casefold()
    rules = (
        ("count", ("bao nhiêu", "mấy ", "how many", "number of")),
        ("color", ("màu gì", "màu nào", "what color", "which color")),
        ("person", ("ai ", "là ai", "who ", "whose")),
        ("location", ("ở đâu", "nơi nào", "where ", "which place")),
        ("time", ("khi nào", "lúc nào", "when ", "what time")),
        ("yes_no", ("có phải", "đúng không", "is there", "does ", "did ")),
        ("text", ("ghi gì", "viết gì", "dòng chữ", "what does", "what is written")),
    )
    for answer_type, markers in rules:
        if any(marker in lowered for marker in markers):
            return answer_type
    return None


def _modality_hints(text: str) -> dict[str, float]:
    lowered = text.casefold()
    hints = {"visual": 1.0}
    if any(
        term in lowered
        for term in ("ghi gì", "viết gì", "dòng chữ", "biển hiệu", "logo", "ocr")
    ):
        hints["ocr"] = 2.0
    if any(
        term in lowered
        for term in ("nói gì", "phát biểu", "nghe thấy", "lời thoại", "âm thanh")
    ):
        hints["asr"] = 2.0
    if any(term in lowered for term in ("tiêu đề", "mô tả video", "kênh", "chương trình")):
        hints["metadata"] = 1.75
    if any(term in lowered for term in ("bao nhiêu", "mấy người", "mấy chiếc")):
        hints["objects"] = 1.5
    return hints


def _split_events(text: str) -> tuple[QueryEvent, ...]:
    numbered = [match.strip(" .") for match in _NUMBERED_EVENT_PATTERN.findall(text)]
    fragments = numbered if len(numbered) >= 2 else _SEQUENCE_PATTERN.split(text)
    fragments = [fragment.strip(" ,.;:\n") for fragment in fragments if fragment.strip()]
    if len(fragments) < 2:
        return ()
    return tuple(
        QueryEvent(order=index, description=fragment)
        for index, fragment in enumerate(fragments, start=1)
    )


def _split_qa_context(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        question = lines[-1]
        context = " ".join(lines[:-1])
        if "?" in question or _expected_answer_type(question):
            return context, question
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) > 1 and _expected_answer_type(sentences[-1]):
        return " ".join(sentences[:-1]).strip(), sentences[-1].strip()
    return text.strip(), text.strip()


class HeuristicQueryParser:
    """Conservative parser that works offline and can later be replaced by an LLM."""

    def parse(
        self, text: str, *, task_type: TaskType | str | None = None
    ) -> ParsedQuery:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            raise ValueError("query text must not be empty")

        explicit_type = TaskType(task_type) if task_type is not None else None
        events = _split_events(text)
        answer_type = _expected_answer_type(cleaned)
        inferred_type = (
            TaskType.TRAKE
            if events
            else TaskType.QA
            if answer_type is not None or cleaned.endswith("?")
            else TaskType.KIS
        )
        resolved_type = explicit_type or inferred_type

        retrieval_text = cleaned
        question: str | None = None
        if resolved_type == TaskType.QA:
            retrieval_text, question = _split_qa_context(text)
        if resolved_type != TaskType.TRAKE:
            events = ()

        return ParsedQuery(
            raw_text=cleaned,
            task_type=resolved_type,
            retrieval_text=retrieval_text or cleaned,
            question=question,
            language=_language(cleaned),
            expected_answer_type=answer_type,
            events=events,
            modality_hints=_modality_hints(cleaned),
        )


class IdentityQueryExpander:
    """Return only the canonical retrieval text."""

    def expand(self, query: ParsedQuery) -> Sequence[QueryVariant]:
        return (QueryVariant(query.retrieval_text),)


class HeuristicQueryExpander:
    """Produce safe variants and optionally delegate translation/paraphrasing.

    Callbacks receive a plain string, keeping all API/model dependencies outside
    the retrieval core. Duplicate normalized texts are removed deterministically.
    """

    def __init__(
        self,
        *,
        translator: Callable[[str], str | None] | None = None,
        paraphraser: Callable[[str], Sequence[str]] | None = None,
        max_paraphrases: int = 3,
    ) -> None:
        self._translator = translator
        self._paraphraser = paraphraser
        self._max_paraphrases = max(0, max_paraphrases)

    def expand(self, query: ParsedQuery) -> Sequence[QueryVariant]:
        variants = [QueryVariant(query.retrieval_text, weight=1.0, kind="original")]
        if self._translator:
            translated = self._translator(query.retrieval_text)
            if translated and translated.strip():
                variants.append(
                    QueryVariant(translated.strip(), weight=0.9, kind="translation")
                )
        if self._paraphraser:
            for text in self._paraphraser(query.retrieval_text)[: self._max_paraphrases]:
                if text and text.strip():
                    variants.append(
                        QueryVariant(text.strip(), weight=0.8, kind="paraphrase")
                    )
        if query.task_type == TaskType.TRAKE:
            variants.extend(
                QueryVariant(event.description, weight=0.85, kind="event")
                for event in query.events
            )

        deduplicated: list[QueryVariant] = []
        seen: set[str] = set()
        for variant in variants:
            key = " ".join(variant.text.casefold().split())
            if key not in seen:
                seen.add(key)
                deduplicated.append(variant)
        return tuple(deduplicated)
