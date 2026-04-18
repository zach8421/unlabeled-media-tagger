"""Tests for fetch stage."""

import pytest
from unlabeled_media_tagger.drive.files import parse_drive_folder_id
from unlabeled_media_tagger.pipeline.fetch import FetchStage


def test_fetch_stage_initialization():
    """Test that FetchStage can be initialized."""
    stage = FetchStage()
    assert stage is not None
    assert isinstance(stage.config, dict)


def test_fetch_stage_with_config():
    """Test that FetchStage accepts configuration."""
    config = {"api_key": "test_key"}
    stage = FetchStage(config=config)
    assert stage.config == config


def test_fetch_requires_location():
    """Test that fetch requires a Drive location."""
    stage = FetchStage()
    with pytest.raises(ValueError):
        stage.fetch()


def test_parse_drive_folder_id_accepts_raw_id():
    """Test parsing a raw folder ID."""
    assert parse_drive_folder_id("abc123") == "abc123"


def test_parse_drive_folder_id_accepts_folder_url():
    """Test parsing a Drive folder URL."""
    url = "https://drive.google.com/drive/folders/folder_123?usp=sharing"
    assert parse_drive_folder_id(url) == "folder_123"
