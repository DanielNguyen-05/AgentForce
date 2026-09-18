from agentforce.retrieval.query import HeuristicQueryExpander, HeuristicQueryParser
from agentforce.retrieval.types import TaskType


def test_parser_detects_qa_context_and_answer_type() -> None:
    parser = HeuristicQueryParser()

    parsed = parser.parse("Cảnh một phụ nữ mặc váy đỏ đang cầm ly.\nChiếc ly màu gì?")

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


def test_parser_supports_official_e_numbered_event_lines_after_context() -> None:
    parsed = HeuristicQueryParser().parse(
        "Cảnh mở đầu không phải event.\n"
        "E1 Hai con rồng vàng xuất hiện đầy đủ.\n"
        "E2 Con lân hoàn tất cú xoay.\n"
        "E3 Dùi chạm vào kẻng đồng.",
        task_type="trake",
    )

    assert [event.description for event in parsed.events] == [
        "Hai con rồng vàng xuất hiện đầy đủ",
        "Con lân hoàn tất cú xoay",
        "Dùi chạm vào kẻng đồng",
    ]


def test_explicit_kis_does_not_parse_sequence_words_as_trake_events() -> None:
    text = (
        "Con lân nhảy qua hai chiếc cột. "
        "Cảnh quay kết thúc khi con lân tiếp tục nhảy sang các cột tiếp theo."
    )

    parsed = HeuristicQueryParser().parse(text, task_type="kis")

    assert parsed.task_type == TaskType.KIS
    assert parsed.retrieval_text == text
    assert parsed.events == ()


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
