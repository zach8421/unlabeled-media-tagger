"""
Placeholder tests for detect stage.

TODO: Add tests when detect stage is implemented.
"""

import pytest
from unlabeled_media_tagger.pipeline.detect import DetectStage


def test_detect_stage_initialization():
    """Test that DetectStage can be initialized."""
    stage = DetectStage()
    assert stage is not None
    assert isinstance(stage.config, dict)


def test_detect_faces_not_implemented():
    """Test that detect_faces method raises NotImplementedError."""
    stage = DetectStage()
    with pytest.raises(NotImplementedError):
        stage.detect_faces("test_image.jpg")


def test_detect_objects_not_implemented():
    """Test that detect_objects method raises NotImplementedError."""
    stage = DetectStage()
    with pytest.raises(NotImplementedError):
        stage.detect_objects("test_image.jpg")
