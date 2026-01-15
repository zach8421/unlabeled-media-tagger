"""
Placeholder tests for compare stage.

TODO: Add tests when compare stage is implemented.
"""

import pytest
from unlabeled_media_tagger.pipeline.compare import CompareStage


def test_compare_stage_initialization():
    """Test that CompareStage can be initialized."""
    stage = CompareStage()
    assert stage is not None
    assert isinstance(stage.config, dict)


def test_compare_faces_not_implemented():
    """Test that compare_faces method raises NotImplementedError."""
    stage = CompareStage()
    with pytest.raises(NotImplementedError):
        stage.compare_faces([])


def test_build_face_database_not_implemented():
    """Test that build_face_database method raises NotImplementedError."""
    stage = CompareStage()
    with pytest.raises(NotImplementedError):
        stage.build_face_database({})
