from __future__ import annotations

import json

from agentforce.preprocessing.objects import load_object_frame


def test_object_normalization_filters_and_suppresses_duplicate_boxes(tmp_path) -> None:
    path = tmp_path / "001.json"
    path.write_text(
        json.dumps(
            {
                "detection_scores": ["0.9", "0.8", "0.1", "0.7"],
                "detection_class_names": ["/m/person", "/m/person", "/m/cat", "/m/bike"],
                "detection_class_entities": ["Person", "Person", "Cat", "Bicycle"],
                "detection_boxes": [
                    ["0.1", "0.1", "0.8", "0.8"],
                    ["0.11", "0.11", "0.79", "0.79"],
                    ["0.2", "0.2", "0.3", "0.3"],
                    ["0.2", "0.5", "0.6", "0.9"],
                ],
            }
        ),
        encoding="utf-8",
    )
    frame = load_object_frame(
        path,
        "L26_V001_K000001",
        min_score=0.2,
        label_aliases_vi={"Person": "người", "Bicycle": "xe đạp"},
    )
    assert frame.counts == {"bicycle": 1, "person": 1}
    assert len(frame.detections) == 2
    assert frame.detections[0].bbox.x_min == 0.1
    assert "person/người" in frame.searchable_text
