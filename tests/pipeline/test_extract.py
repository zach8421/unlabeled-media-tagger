"""Tests for extract stage."""

import pytest
from unlabeled_media_tagger.pipeline.extract import ExtractStage


def test_extract_stage_initialization():
    """Test that ExtractStage can be initialized."""
    stage = ExtractStage()
    assert stage is not None
    assert isinstance(stage.config, dict)


def test_extract_missing_file():
    """Test that extract raises for a missing media file."""
    stage = ExtractStage()
    with pytest.raises(FileNotFoundError):
        stage.extract("test_file.mp4")


def test_extract_image_returns_single_frame():
    """Test that images are represented as a single frame."""
    stage = ExtractStage()
    result = stage.extract("tests/assets/sample_image.jpg")

    assert result["media_type"] == "image"
    assert result["frames"] == [
        {
            "path": "tests/assets/sample_image.jpg",
            "timestamp_sec": None,
            "frame_index": None,
        }
    ]
