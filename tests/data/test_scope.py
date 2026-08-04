from __future__ import annotations

import pytest

from agentforce.data.scope import (
    assert_exact_scope,
    filter_manifest,
    normalize_video_ids,
    scope_hash,
    select_videos,
)
from agentforce.data.schemas import DatasetManifest, ValidationIssue, VideoRecord


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_root="/dataset",
        videos=[
            VideoRecord(video_id="L21_V001", collection="L21"),
            VideoRecord(video_id="L21_V002", collection="L21"),
            VideoRecord(video_id="L22_V001", collection="L22"),
        ],
        issues=[
            ValidationIssue("warning", "x", "one", video_id="L21_V001"),
            ValidationIssue("warning", "x", "other", video_id="L22_V001"),
        ],
    )


def test_filter_manifest_preserves_requested_order_and_relevant_issues() -> None:
    scoped = filter_manifest(_manifest(), ["l21_v002", "L21_V001"])
    assert [video.video_id for video in scoped.videos] == ["L21_V002", "L21_V001"]
    assert [issue.video_id for issue in scoped.issues] == ["L21_V001"]


def test_select_videos_cannot_escape_configured_scope() -> None:
    with pytest.raises(ValueError, match="outside the configured scope"):
        select_videos(
            _manifest(),
            configured_ids=["L21_V001", "L21_V002"],
            requested_ids=["L22_V001"],
        )


def test_scope_hash_is_order_independent_and_ids_are_deduplicated() -> None:
    assert normalize_video_ids(["l21_v001", "L21_V001"]) == ("L21_V001",)
    assert scope_hash(["L21_V001", "L21_V002"]) == scope_hash(
        ["L21_V002", "L21_V001"]
    )
    assert_exact_scope(["L21_V002", "L21_V001"], ["L21_V001", "L21_V002"], label="x")
