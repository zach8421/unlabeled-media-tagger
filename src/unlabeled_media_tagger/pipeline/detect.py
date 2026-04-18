"""Detect Stage - computer vision and facial recognition."""

from __future__ import annotations


class DetectStage:
    """
    Detect stage for running computer vision and facial recognition models.
    
    This stage is responsible for:
    - Running facial detection models on images/frames
    - Running object detection models
    - Extracting facial embeddings for comparison
    - Detecting other visual features (scenes, text, etc.)
    """
    
    def __init__(self, config=None):
        """
        Initialize the detect stage.
        
        Args:
            config: Configuration dictionary for model settings
        """
        self.config = config or {}
    
    def detect_faces(self, image):
        """
        Detect faces in an image.
        
        Args:
            image: Image data or path to image file
            
        Returns:
            List of detected face bounding boxes and embeddings
            
        Raises:
            FileNotFoundError: If image path does not exist.
            ValueError: If image cannot be processed.
        """
        from unlabeled_media_tagger.pipeline.embed_faces import embed_faces_in_image

        return embed_faces_in_image(
            image,
            detector_backend=self.config.get("detector_backend", "retinaface"),
            model_name=self.config.get("model_name", "ArcFace"),
        )
    
    def detect_objects(self, image):
        """
        Detect objects in an image.
        
        Args:
            image: Image data or path to image file
            
        Returns:
            List of detected objects with labels and confidence scores
            
        Raises:
            NotImplementedError: Object detection is not part of the current pipeline.
        """
        raise NotImplementedError("Object detection is not implemented")

    def process_extracted_media(self, extracted_media: dict, media_record: dict) -> list[dict]:
        """
        Extract face embeddings for every frame/image in an extracted media record.

        Args:
            extracted_media: Output from ExtractStage.extract.
            media_record: Source Drive/local media record.

        Returns:
            List of normalized face records.
        """
        face_records = []
        for frame in extracted_media["frames"]:
            faces = self.detect_faces(frame["path"])
            for face_index, face in enumerate(faces):
                face_records.append(
                    {
                        **face,
                        "drive_id": media_record.get("drive_id"),
                        "media_name": media_record.get("name"),
                        "media_path": media_record.get("local_path"),
                        "frame_path": frame["path"],
                        "timestamp_sec": frame.get("timestamp_sec"),
                        "frame_index": frame.get("frame_index"),
                        "face_index": face_index,
                    }
                )

        return face_records
