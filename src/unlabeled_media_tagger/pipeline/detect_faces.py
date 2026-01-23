"""
Face Detection Module - DeepFace Integration

This module provides face detection functionality using DeepFace library.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from deepface import DeepFace


def detect_faces_in_image(
    image_path: str,
    detector_backend: str = "retinaface"
) -> List[Dict]:
    """
    Detect faces in an image using DeepFace.
    
    Args:
        image_path: Path to the image file
        detector_backend: Detection backend to use (default: "retinaface")
                         Options: retinaface, mtcnn, opencv, ssd, dlib, mediapipe
    
    Returns:
        List of detected faces, each containing:
            - bbox: dict with x, y, w, h (bounding box coordinates)
            - confidence: float (detection confidence score)
    
    Raises:
        FileNotFoundError: If image_path does not exist
        ValueError: If image cannot be processed
    """
    # Verify image exists
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Run face detection
    try:
        faces = DeepFace.extract_faces(
            img_path=str(image_path),
            detector_backend=detector_backend,
            enforce_detection=False,  # Don't raise error if no faces found
            align=False  # Don't align faces, just detect
        )
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")
    
    # Normalize results
    results = []
    for face in faces:
        # Extract facial area (bounding box)
        facial_area = face.get('facial_area', {})
        
        # Get confidence score
        confidence = face.get('confidence', 0.0)
        
        # Build normalized result
        result = {
            'bbox': {
                'x': facial_area.get('x', 0),
                'y': facial_area.get('y', 0),
                'w': facial_area.get('w', 0),
                'h': facial_area.get('h', 0)
            },
            'confidence': confidence
        }
        results.append(result)
    
    return results


def main():
    """CLI entry point for face detection."""
    if len(sys.argv) < 2:
        print("Usage: python -m unlabeled_media_tagger.pipeline.detect_faces <image_path> [detector_backend]")
        print("\nDetector backends: retinaface (default), mtcnn, opencv, ssd, dlib, mediapipe")
        sys.exit(1)
    
    image_path = sys.argv[1]
    detector_backend = sys.argv[2] if len(sys.argv) > 2 else "retinaface"
    
    print(f"Detecting faces in: {image_path}")
    print(f"Using detector: {detector_backend}\n")
    
    try:
        results = detect_faces_in_image(image_path, detector_backend)
        
        print(f"Detected {len(results)} face(s):\n")
        
        for i, face in enumerate(results, 1):
            bbox = face['bbox']
            confidence = face['confidence']
            print(f"Face {i}:")
            print(f"  Bounding Box: x={bbox['x']}, y={bbox['y']}, w={bbox['w']}, h={bbox['h']}")
            print(f"  Confidence: {confidence:.4f}")
            print()
        
        if len(results) == 0:
            print("No faces detected in the image.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
