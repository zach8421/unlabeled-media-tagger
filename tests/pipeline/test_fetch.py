"""
Placeholder tests for fetch stage.

TODO: Add tests when fetch stage is implemented.
"""

import pytest
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


def test_fetch_not_implemented():
    """Test that fetch method raises NotImplementedError."""
    stage = FetchStage()
    with pytest.raises(NotImplementedError):
        stage.fetch()
