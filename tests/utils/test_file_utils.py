"""
Tests for utility modules.
"""

from unlabeled_media_tagger.utils.file_utils import (
    is_image_file,
    is_video_file,
)


def test_is_image_file():
    """Test image file detection."""
    assert is_image_file("photo.jpg") is True
    assert is_image_file("photo.jpeg") is True
    assert is_image_file("photo.png") is True
    assert is_image_file("photo.gif") is True
    assert is_image_file("photo.bmp") is True
    assert is_image_file("photo.webp") is True
    assert is_image_file("video.mp4") is False
    assert is_image_file("document.pdf") is False


def test_is_video_file():
    """Test video file detection."""
    assert is_video_file("video.mp4") is True
    assert is_video_file("video.avi") is True
    assert is_video_file("video.mov") is True
    assert is_video_file("video.mkv") is True
    assert is_video_file("video.wmv") is True
    assert is_video_file("video.flv") is True
    assert is_video_file("photo.jpg") is False
    assert is_video_file("document.pdf") is False
