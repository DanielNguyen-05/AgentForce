from agentforce.retrieval.query import HeuristicQueryExpander, HeuristicQueryParser
from agentforce.retrieval.types import TaskType


def test_parser_detects_qa_context_and_answer_type() -> None:
    parser = HeuristicQueryParser()

    parsed = parser.parse(
        "Cảnh một phụ nữ mặc váy đỏ đang cầm ly.\nChiếc ly màu gì?"
    )

    assert parsed.task_type == TaskType.QA
    assert parsed.retrieval_text.startswith("Cảnh một phụ nữ")
    assert parsed.question == "Chiếc ly màu gì?"
    assert parsed.expected_answer_type == "color"
    assert parsed.language == "vi"


def test_parser_detects_ordered_events_from_connectors() -> None:
    parsed = HeuristicQueryParser().parse(
        "Vận động viên giậm nhảy, sau đó bay qua xà, tiếp theo tiếp đất"
    )

    assert parsed.task_type == TaskType.TRAKE
    assert [event.description for event in parsed.events] == [
        "Vận động viên giậm nhảy",
        "bay qua xà",
        "tiếp đất",
    ]


def test_parser_supports_numbered_event_lines() -> None:
    parsed = HeuristicQueryParser().parse("1. mở cửa\n2. bước vào\n3. ngồi xuống")
    assert [event.order for event in parsed.events] == [1, 2, 3]


def test_expander_deduplicates_callbacks() -> None:
    parsed = HeuristicQueryParser().parse("người đang đi xe đạp", task_type="kis")
    expander = HeuristicQueryExpander(
        translator=lambda _: "a person riding a bicycle",
        paraphraser=lambda _: ["A PERSON RIDING A BICYCLE", "cyclist on a road"],
    )

    variants = expander.expand(parsed)

    assert [variant.kind for variant in variants] == [
        "original",
        "translation",
        "paraphrase",
    ]

