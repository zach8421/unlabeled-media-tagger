"""
File system utilities for media file handling.
"""

from pathlib import Path
from typing import List


SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}


def is_image_file(file_path: str) -> bool:
    """
    Check if a file is a supported image format.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a supported image format
    """
    return Path(file_path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_video_file(file_path: str) -> bool:
    """
    Check if a file is a supported video format.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a supported video format
    """
    return Path(file_path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def get_media_files(directory: str) -> List[str]:
    """
    Get all supported media files from a directory.
    
    Args:
        directory: Path to the directory to scan
        
    Returns:
        List of paths to media files
        
    Raises:
        NotImplementedError: This is a placeholder for future implementation
    """
    raise NotImplementedError("Media file scanning not yet implemented")
