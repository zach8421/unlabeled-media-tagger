"""Tests for enrich stage."""

import pytest
from unlabeled_media_tagger.pipeline.enrich import (
    DESCRIPTION_END,
    DESCRIPTION_START,
    EnrichStage,
    build_description,
    build_drive_metadata,
)


def test_enrich_stage_initialization():
    """Test that EnrichStage can be initialized."""
    stage = EnrichStage()
    assert stage is not None
    assert isinstance(stage.config, dict)


def test_enrich_local_not_implemented():
    """Test that enrich_local method raises NotImplementedError."""
    stage = EnrichStage()
    with pytest.raises(NotImplementedError):
        stage.enrich_local("test_file.mp4", {})


def test_enrich_drive_requires_service():
    """Test that enrich_drive requires a Drive service."""
    stage = EnrichStage()
    with pytest.raises(ValueError):
        stage.enrich_drive("file_id", {})


def test_build_description_preserves_human_text_and_replaces_block():
    """Test that managed metadata block replacement is stable."""
    first = build_description("human text", {"face_count": 1})
    second = build_description(first, {"face_count": 2})

    assert second.startswith("human text")
    assert second.count(DESCRIPTION_START) == 1
    assert second.count(DESCRIPTION_END) == 1
    assert '"face_count":2' in second


def test_build_drive_metadata_lists_clusters():
    """Test compact Drive metadata payload."""
    metadata = build_drive_metadata(
        [
            {"cluster_label": "person_001"},
            {"cluster_label": "person_000"},
            {"cluster_label": "person_001"},
        ]
    )

    assert metadata["schema"] == "unlabeled-media-tagger"
    assert metadata["face_count"] == 3
    assert metadata["clusters"] == ["person_000", "person_001"]
