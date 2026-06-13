"""Tests for extract stage."""

import pytest
from unlabeled_media_tagger.pipeline.extract import (
    ExtractStage,
    UnreadableMediaError,
)


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


def test_extract_unsupported_type_raises_unreadable(tmp_path):
    """An existing file with an unsupported extension raises UnreadableMediaError."""
    bad = tmp_path / "raw.braw"
    bad.write_bytes(b"\x00\x01\x02")
    stage = ExtractStage()
    with pytest.raises(UnreadableMediaError):
        stage.extract(str(bad))


def test_extract_corrupt_video_raises_unreadable(tmp_path):
    """A file that looks like a video but can't be opened is UnreadableMediaError."""
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"not really an mp4")
    stage = ExtractStage()
    with pytest.raises(UnreadableMediaError):
        stage.extract(str(bad))


def test_unreadable_is_value_error_subclass():
    """Backward compat: callers catching ValueError still catch this."""
    assert issubclass(UnreadableMediaError, ValueError)


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
