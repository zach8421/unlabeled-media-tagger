"""
Tests for configuration settings.
"""

from unlabeled_media_tagger.config.settings import (
    Config,
    GoogleDriveConfig,
    ModelConfig,
    PipelineConfig,
)


def test_config_initialization():
    """Test that Config can be initialized."""
    config = Config()
    assert config is not None
    assert isinstance(config.google_drive, GoogleDriveConfig)
    assert isinstance(config.models, ModelConfig)
    assert isinstance(config.pipeline, PipelineConfig)


def test_google_drive_config():
    """Test GoogleDriveConfig initialization."""
    config = GoogleDriveConfig()
    assert config.credentials_path is None
    assert config.token_path is None
    assert config.scopes == []
    assert config.folder_id is None


def test_model_config():
    """Test ModelConfig initialization."""
    config = ModelConfig()
    assert config.face_detection_model is None
    assert config.face_recognition_model is None
    assert config.object_detection_model is None
    assert config.detection_threshold == 0.7
    assert config.device == "cpu"


def test_pipeline_config():
    """Test PipelineConfig initialization."""
    config = PipelineConfig()
    assert config.batch_size == 10
    assert config.frame_interval == 1.0
    assert config.max_frames == 100
    assert config.output_dir == "./output"
