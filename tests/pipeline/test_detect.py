"""Tests for detect stage."""

import pytest
from unlabeled_media_tagger.pipeline.detect import DetectStage


def test_detect_stage_initialization():
    """Test that DetectStage can be initialized."""
    stage = DetectStage()
    assert stage is not None
    assert isinstance(stage.config, dict)


def test_detect_objects_not_implemented():
    """Test that detect_objects method raises NotImplementedError."""
    stage = DetectStage()
    with pytest.raises(NotImplementedError):
        stage.detect_objects("test_image.jpg")


def test_process_extracted_media_annotates_face_records(monkeypatch):
    """Test that frame-level detections are normalized with media metadata."""
    stage = DetectStage()

    def fake_detect_faces(image):
        return [
            {
                "source_path": image,
                "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                "confidence": 0.9,
                "embedding": [1.0, 0.0],
            }
        ]

    monkeypatch.setattr(stage, "detect_faces", fake_detect_faces)
    records = stage.process_extracted_media(
        {
            "frames": [
                {
                    "path": "frame.jpg",
                    "timestamp_sec": 1.5,
                    "frame_index": 10,
                }
            ]
        },
        {
            "drive_id": "drive-1",
            "name": "video.mp4",
            "local_path": "downloads/video.mp4",
        },
    )

    assert records[0]["drive_id"] == "drive-1"
    assert records[0]["media_name"] == "video.mp4"
    assert records[0]["frame_path"] == "frame.jpg"
    assert records[0]["face_index"] == 0
