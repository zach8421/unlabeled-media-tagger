"""
Face Embedding Extraction Module

This module extracts face embeddings using DeepFace for face recognition and similarity matching.
"""

import logging
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np
from deepface import DeepFace

from .detect_faces import detect_faces_in_image

logger = logging.getLogger(__name__)


def embed_faces_in_image(
    image_path: str,
    detector_backend: str = "retinaface",
    model_name: str = "ArcFace",
) -> List[Dict]:
    """
    Detect faces in an image and extract embeddings for each face.
    
    Args:
        image_path: Path to the image file
        detector_backend: Detection backend to use (default: "retinaface")
                         Options: retinaface, mtcnn, opencv, ssd, dlib, mediapipe
        model_name: Embedding model to use (default: "ArcFace")
                   Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace
    
    Returns:
        List of face records, each containing:
            - source_path: str (path to source image)
            - bbox: dict with x, y, w, h (original bounding box coordinates)
            - bbox_clipped: dict with x, y, w, h (clipped bounding box coordinates)
            - confidence: float (detection confidence score)
            - embedding: list[float] (face embedding vector)
            - model_name: str (name of embedding model used)
            - detector_backend: str (name of detector backend used)
    
    Raises:
        FileNotFoundError: If image_path does not exist
        ValueError: If image cannot be processed
    """
    # Verify image exists
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image with OpenCV
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    height, width = image.shape[:2]
    
    # Detect faces
    try:
        detections = detect_faces_in_image(image_path, detector_backend)
    except Exception as e:
        raise ValueError(f"Error detecting faces: {e}")
    
    # Track statistics for debugging
    faces_total_detections = len(detections)
    faces_invalid_bbox = 0
    faces_too_small = 0
    faces_embed_failed = 0
    faces_embedded_success = 0
    
    # Extract embeddings for each detected face
    results = []
    for detection in detections:
        bbox = detection['bbox']
        # Ensure bbox values are integers for safe array slicing
        x = int(bbox['x'])
        y = int(bbox['y'])
        w = int(bbox['w'])
        h = int(bbox['h'])
        
        # Clip bbox to image bounds (using int operations)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)
        
        # Skip if crop is invalid
        if x2 <= x1 or y2 <= y1:
            faces_invalid_bbox += 1
            logger.debug(f"Skipped face (invalid bbox after clipping): x={x}, y={y}, w={w}, h={h}")
            continue
        
        # Crop face region
        face_crop = image[y1:y2, x1:x2]
        
        # Skip if crop is too small
        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            faces_too_small += 1
            logger.debug(f"Skipped face (too small): crop_shape={face_crop.shape}, bbox=({x},{y},{w},{h})")
            continue
        
        # Generate embedding
        try:
            # DeepFace.represent expects RGB, OpenCV loads as BGR
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Use DeepFace to extract embedding
            # detector_backend="skip" tells DeepFace not to re-detect faces
            embedding_result = DeepFace.represent(
                img_path=face_crop_rgb,
                model_name=model_name,
                detector_backend="skip",
                enforce_detection=False
            )
            
            # Extract embedding vector
            # DeepFace.represent returns a list of dicts, take first result
            if isinstance(embedding_result, list) and len(embedding_result) > 0:
                embedding = embedding_result[0].get('embedding', [])
            else:
                continue
            
            # Normalize to list[float]
            embedding = [float(x) for x in embedding]
            
            # Build result record
            result = {
                'source_path': str(image_path),
                'bbox': bbox,
                'bbox_clipped': {
                    'x': x1,
                    'y': y1,
                    'w': x2 - x1,
                    'h': y2 - y1
                },
                'confidence': detection['confidence'],
                'embedding': embedding,
                'model_name': model_name,
                'detector_backend': detector_backend
            }
            results.append(result)
            faces_embedded_success += 1
            
        except Exception as e:
            # Skip faces that fail embedding extraction
            # This can happen if the crop is too small or unclear
            faces_embed_failed += 1
            logger.debug(f"Skipped face (embedding failed): {e}, bbox=({x},{y},{w},{h})")
            continue
    
    # Log summary if debug enabled
    logger.debug(
        f"Embedding extraction complete: {faces_embedded_success}/{faces_total_detections} successful, "
        f"{faces_invalid_bbox} invalid bbox, {faces_too_small} too small, {faces_embed_failed} embed failed"
    )
    
    return results
