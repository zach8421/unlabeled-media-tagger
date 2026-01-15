"""
Placeholder tests for enrich stage.

TODO: Add tests when enrich stage is implemented.
"""

import pytest
from unlabeled_media_tagger.pipeline.enrich import EnrichStage


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


def test_enrich_drive_not_implemented():
    """Test that enrich_drive method raises NotImplementedError."""
    stage = EnrichStage()
    with pytest.raises(NotImplementedError):
        stage.enrich_drive("file_id", {})
