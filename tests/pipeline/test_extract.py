"""
Placeholder tests for extract stage.

TODO: Add tests when extract stage is implemented.
"""

import pytest
from unlabeled_media_tagger.pipeline.extract import ExtractStage


def test_extract_stage_initialization():
    """Test that ExtractStage can be initialized."""
    stage = ExtractStage()
    assert stage is not None
    assert isinstance(stage.config, dict)


def test_extract_not_implemented():
    """Test that extract method raises NotImplementedError."""
    stage = ExtractStage()
    with pytest.raises(NotImplementedError):
        stage.extract("test_file.mp4")
