"""
Configuration settings for the unlabeled-media-tagger application.

This module defines configuration structure for all pipeline stages.
"""


class Config:
    """
    Main configuration class for the application.
    
    This class holds configuration for:
    - Google Drive API credentials and settings
    - Computer vision model paths and parameters
    - Pipeline stage settings
    - Processing options and thresholds
    """
    
    def __init__(self):
        """Initialize configuration with default values."""
        self.google_drive = GoogleDriveConfig()
        self.models = ModelConfig()
        self.pipeline = PipelineConfig()
    
    @classmethod
    def from_file(cls, config_path):
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Config instance
            
        Raises:
            NotImplementedError: This is a placeholder for future implementation
        """
        raise NotImplementedError("Configuration loading not yet implemented")


class GoogleDriveConfig:
    """Configuration for Google Drive API integration."""
    
    def __init__(self):
        """Initialize Google Drive configuration."""
        self.credentials_path = None  # Path to credentials.json
        self.token_path = None  # Path to token storage
        self.scopes = []  # API scopes needed
        self.folder_id = None  # Target folder ID to process


class ModelConfig:
    """Configuration for computer vision and ML models."""
    
    def __init__(self):
        """Initialize model configuration."""
        self.face_detection_model = None  # Path to face detection model
        self.face_recognition_model = None  # Path to face recognition model
        self.object_detection_model = None  # Path to object detection model
        self.detection_threshold = 0.7  # Confidence threshold for detections
        self.device = "cpu"  # Device to run models on (cpu, cuda, mps)


class PipelineConfig:
    """Configuration for pipeline processing options."""
    
    def __init__(self):
        """Initialize pipeline configuration."""
        self.batch_size = 10  # Number of files to process in a batch
        self.frame_interval = 1.0  # Seconds between frame extractions for videos
        self.max_frames = 100  # Maximum frames to extract per video
        self.output_dir = "./output"  # Directory for processed outputs
